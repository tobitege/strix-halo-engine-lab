param(
    [Parameter(Mandatory=$true)][string]$BuildDir,
    [Parameter(Mandatory=$true)][string]$Weights,
    [ValidateRange(1,3)][int]$Iterations=1,
    [switch]$MmaIndex,
    [ValidateSet('fp32','mma','mma-proj-fp32-score','fp32-proj-mma-score','tf32x3')][string]$IndexVariant='fp32',
    [switch]$VariantSweep,
    [switch]$PlanOnly
)
# One aggregate deadline including SHA verification, launches, warmups, every
# full FP32 reference and readback. Timeout is failure; no partial success.
$ErrorActionPreference='Stop'
$deadline=[DateTime]::UtcNow.AddSeconds(570)
$build=(Resolve-Path $BuildDir).Path
$weightsPath=(Resolve-Path $Weights).Path
$manifest=Get-Content ($weightsPath+'.json') -Raw | ConvertFrom-Json
if ($manifest.format -ne 'GLMATT01' -or $manifest.layer -ne 7 -or
    $manifest.layer_type -ne 'deepseek_sparse_attention' -or
    (Get-FileHash $weightsPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.sha256) {
    throw 'Real layer-7 DSA archive type/layer/SHA256 differs from manifest'
}
$exe=Join-Path $build 'glm_dsa_bench.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw 'Build the DSA benchmark first' }
if ($MmaIndex) { $IndexVariant='mma' }
$identities=@{
    'fp32'='glm-pf-rtx-native-v1-dsa64-f32idx'
    'mma'='glm-pf-rtx-native-v1-dsa64-mmaidx'
    'mma-proj-fp32-score'='glm-pf-rtx-native-v2-dsa64-mmap-f32s'
    'fp32-proj-mma-score'='glm-pf-rtx-native-v2-dsa64-f32p-mmas'
    'tf32x3'='glm-pf-rtx-native-v2-dsa64-tf32x3p-bf16x3s'
}
$cases=@()
if ($VariantSweep) {
    # Provisional accurate candidate at 1K; all arithmetic variants at 8K/32K.
    $cases+= [PSCustomObject]@{rows=1024;variant='tf32x3'}
    foreach ($rows in @(8192,32768)) {
        foreach ($variant in @('fp32','mma','mma-proj-fp32-score','fp32-proj-mma-score','tf32x3')) {
            $cases+= [PSCustomObject]@{rows=$rows;variant=$variant}
        }
    }
} else { foreach ($rows in @(1024,8192,32768)) { $cases+= [PSCustomObject]@{rows=$rows;variant=$IndexVariant} } }
$runs=@($cases | ForEach-Object {
    $extra=if ($MmaIndex -and -not $VariantSweep) {@('--mma-index')} else {@('--index-variant',$_.variant)}
    [PSCustomObject]@{rows=$_.rows;variant=$_.variant;identity=$identities[$_.variant];arguments=(@('--weights',('"'+$weightsPath+'"'),'--rows',$_.rows,'--iters',$Iterations,'--allow-gpu')+$extra)}
})
$plan=[PSCustomObject]@{status='PLANNED';aggregate_timeout_seconds=570;kill_wait_seconds=5;identity=$(if ($VariantSweep) {'VARIANT_SWEEP_DRAFT'} else {$identities[$IndexVariant]});
    executable_sha256=(Get-FileHash $exe -Algorithm SHA256).Hash.ToLowerInvariant();archive_sha256=$manifest.sha256;
    near_tie_abs=0.001;near_tie_rel=0.001;runs=$runs}
$plan | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $build 'dsa-window-plan.json')
if ($PlanOnly) { $plan | ConvertTo-Json -Depth 6;exit 0 }
if ($env:CUDA_DEVICE_ORDER -ne 'PCI_BUS_ID' -or $env:CUDA_VISIBLE_DEVICES -notmatch '^GPU-[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$') {
    throw 'Set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES to exactly one PRO UUID'
}
$results=@()
foreach ($run in $runs) {
    $left=[int][Math]::Floor(($deadline-[DateTime]::UtcNow).TotalMilliseconds)
    if ($left -le 0) { throw '570-second aggregate deadline; incomplete is not a pass' }
    $stdout=Join-Path $build $(if ($VariantSweep) {"dsa-window-$($run.variant)-$($run.rows).log"} else {"dsa-window-$($run.rows).log"})
    $stderr=Join-Path $build $(if ($VariantSweep) {"dsa-window-$($run.variant)-$($run.rows).err"} else {"dsa-window-$($run.rows).err"})
    $p=Start-Process -FilePath $exe -ArgumentList $run.arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $null=$p.Handle  # Windows PowerShell 5: ExitCode stays $null unless the handle is opened before exit
    if (-not $p.WaitForExit($left)) {
        $p.Kill()
        if (-not $p.WaitForExit(5000)) { throw 'Benchmark kill did not complete' }
        throw '570-second aggregate deadline; partial measurements are not a pass'
    }
    $p.Refresh()
    $log=Get-Content $stdout -Raw
    $selection=[regex]::Match($log,'selection differing_rows=(\d+) differing_pool_memberships=(\d+) near_tie_pool_memberships=(\d+) outside_tolerance_pool_memberships=(\d+)')
    $latency=[regex]::Match($log,'workspace_bytes=\d+ device_ms=([0-9.]+)')
    $counts=if ($selection.Success) {
        [PSCustomObject]@{differing_rows=[long]$selection.Groups[1].Value;differing_pool_memberships=[long]$selection.Groups[2].Value;near_tie_pool_memberships=[long]$selection.Groups[3].Value;outside_tolerance_pool_memberships=[long]$selection.Groups[4].Value}
    } else { $null }
    $pairs=@{}
    foreach ($pair in @('variant_vs_node_mimic','node_mimic_vs_scalar_fp32')) {
        $m=[regex]::Match($log,('selection_pair='+$pair+' differing_rows=(\d+) differing_pool_memberships=(\d+) near_tie_pool_memberships=(\d+) outside_tolerance_pool_memberships=(\d+)'))
        $pairs[$pair]=if ($m.Success) {@{measurement='RECORDED';differing_rows=[long]$m.Groups[1].Value;differing_pool_memberships=[long]$m.Groups[2].Value;near_tie_pool_memberships=[long]$m.Groups[3].Value;outside_tolerance_pool_memberships=[long]$m.Groups[4].Value}} else {@{measurement='NO-REFERENCE'}}
    }
    $missing=@($pairs.Values | Where-Object {$_.measurement -eq 'NO-REFERENCE'}).Count -gt 0
    $results += [PSCustomObject]@{reference_pairs=$pairs;rows=$run.rows;variant=$run.variant;identity=$run.identity;selection=$counts;selection_measurement=$(if ($selection.Success) {'RECORDED'} else {'NO-REFERENCE'});device_ms=$(if ($latency.Success) {[double]$latency.Groups[1].Value} else {$null});exit_code=$p.ExitCode;status=$(if ($missing) {'NO-REFERENCE'} elseif ($p.ExitCode -eq 0) {'PASS_WITH_STATED_TIE_TOLERANCE'} elseif ($p.ExitCode -eq 3) {'SELECTION_MISMATCH'} else {'ERROR'})}
    $results | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $build 'dsa-window-results.json')
    if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3) { throw "DSA benchmark failed; inspect $stdout and $stderr" }
    if ($missing) { throw 'Missing node-mimic pair diagnostics; NO-REFERENCE is not a pass' }
    Get-Content $stdout
}
if (@($results | Where-Object {$_.exit_code -eq 3}).Count) { Write-Output 'DSA diagnostics complete with selection mismatches; see dsa-window-results.json';exit 3 }
Write-Output 'DSA diagnostics complete; endpoint and MTP device qualification remain pending'
