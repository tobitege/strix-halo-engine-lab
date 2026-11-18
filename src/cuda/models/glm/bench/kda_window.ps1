param(
    [Parameter(Mandatory=$true)][string]$BuildDir,
    [Parameter(Mandatory=$true)][string]$Weights,
    [ValidateRange(1,3)][int]$Iterations = 3,
    [switch]$PlanOnly
)
# Explicit GPU launcher, never called by build/tests. The SINGLE aggregate
# deadline covers all three lengths, warmups, complete reference and readbacks.
$ErrorActionPreference = 'Stop'
$deadline = [DateTime]::UtcNow.AddSeconds(570)
$build = (Resolve-Path $BuildDir).Path
$weightsPath = (Resolve-Path $Weights).Path
$manifest = Get-Content ($weightsPath + '.json') -Raw | ConvertFrom-Json
if ($manifest.format -ne 'GLMATT01' -or $manifest.layer_type -ne 'linear_attention' -or
    (Get-FileHash $weightsPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.sha256) {
    throw 'KDA archive format/type/SHA256 differs from export manifest'
}
$exe = Join-Path $build 'glm_kda_bench.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw 'Build the KDA benchmark first' }
$runs = @(1024,8192,32768) | ForEach-Object {
    [PSCustomObject]@{ rows = $_; arguments = @('--weights',('"' + $weightsPath + '"'),'--rows',$_,'--iters',$Iterations,'--allow-gpu') }
}
$plan = [PSCustomObject]@{ status='PLANNED'; aggregate_timeout_seconds=570; identity='glm-pf-rtx-native-v1-kda32';
    executable_sha256=(Get-FileHash $exe -Algorithm SHA256).Hash.ToLowerInvariant(); archive_sha256=$manifest.sha256; runs=$runs }
$plan | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $build 'kda-window-plan.json')
if ($PlanOnly) { $plan | ConvertTo-Json -Depth 5; exit 0 }
if ($env:CUDA_DEVICE_ORDER -ne 'PCI_BUS_ID' -or $env:CUDA_VISIBLE_DEVICES -notmatch '^GPU-[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$') {
    throw 'Set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES to exactly one PRO UUID'
}
foreach ($run in $runs) {
    $left = [int][Math]::Floor(($deadline - [DateTime]::UtcNow).TotalMilliseconds)
    if ($left -le 0) { throw '570-second aggregate deadline; incomplete is not a pass' }
    $stdout = Join-Path $build "kda-window-$($run.rows).log"
    $stderr = Join-Path $build "kda-window-$($run.rows).err"
    $p = Start-Process -FilePath $exe -ArgumentList $run.arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $null = $p.Handle  # Windows PowerShell 5: ExitCode stays $null unless the handle is opened before exit
    if (-not $p.WaitForExit($left)) {
        $p.Kill()
        if (-not $p.WaitForExit(5000)) { throw 'Benchmark kill did not complete' }
        throw '570-second aggregate deadline; partial measurements are not a pass'
    }
    $p.Refresh()
    if ($p.ExitCode -ne 0) { throw "Benchmark failed; inspect $stderr" }
    Get-Content $stdout
}
Write-Output 'KDA timing/accuracy diagnostics complete; statistical endpoint qualification remains pending'
