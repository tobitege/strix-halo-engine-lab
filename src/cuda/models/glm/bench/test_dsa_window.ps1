param([Parameter(Mandatory=$true)][string]$BuildDir)
# CPU-only integration test: the child is a tiny managed executable, never CUDA.
$ErrorActionPreference='Stop'
$root=Join-Path (Resolve-Path $BuildDir).Path ('dsa-launcher-test-'+[Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $root | Out-Null
$fake=Join-Path $root 'glm_dsa_bench.exe'
Add-Type -TypeDefinition @"
using System;
public class DsaLauncherFixture {
    public static int Main(string[] args) {
        string rows="";
        for(int i=0;i+1<args.Length;i++) if(args[i]=="--rows") rows=args[i+1];
        Console.WriteLine("fixture rows="+rows);
        Console.WriteLine("fixture workspace_bytes=0 device_ms=12.5");
        Console.WriteLine("selection differing_rows=1 differing_pool_memberships=2 near_tie_pool_memberships=0 outside_tolerance_pool_memberships=2");
        if(Environment.GetEnvironmentVariable("STRIX_DSA_TEST_NO_REFERENCE")!="1") {
        Console.WriteLine("selection_pair=variant_vs_node_mimic differing_rows=1 differing_pool_memberships=2 near_tie_pool_memberships=0 outside_tolerance_pool_memberships=2");
        Console.WriteLine("selection_pair=node_mimic_vs_scalar_fp32 differing_rows=0 differing_pool_memberships=0 near_tie_pool_memberships=0 outside_tolerance_pool_memberships=0");
        }
        return rows==Environment.GetEnvironmentVariable("STRIX_DSA_TEST_ROW")
            ? int.Parse(Environment.GetEnvironmentVariable("STRIX_DSA_TEST_CODE")) : 0;
    }
}
"@ -OutputAssembly $fake -OutputType ConsoleApplication
$weights=Join-Path $root 'fixture.native'
[IO.File]::WriteAllBytes($weights,[byte[]]@(1,2,3,4))
@{format='GLMATT01';layer=7;layer_type='deepseek_sparse_attention';sha256=(Get-FileHash $weights -Algorithm SHA256).Hash.ToLowerInvariant()} | ConvertTo-Json | Set-Content ($weights+'.json')
$launcher=Join-Path $PSScriptRoot 'dsa_window.ps1'
$oldOrder=$env:CUDA_DEVICE_ORDER;$oldVisible=$env:CUDA_VISIBLE_DEVICES
try {
    # A syntactically valid placeholder UUID; only the managed fixture runs.
    $env:CUDA_DEVICE_ORDER='PCI_BUS_ID';$env:CUDA_VISIBLE_DEVICES='GPU-00000000-0000-0000-0000-000000000000'
    $env:STRIX_DSA_TEST_ROW='8192';$env:STRIX_DSA_TEST_CODE='3'
    & powershell -NoProfile -ExecutionPolicy Bypass -File $launcher -BuildDir $root -Weights $weights | Set-Content (Join-Path $root 'continue.log')
    if ($LASTEXITCODE -ne 3) {throw 'Mismatch suite must return 3'}
    $r=(Get-Content (Join-Path $root 'dsa-window-results.json') -Raw | ConvertFrom-Json)
    if (@($r).Count -ne 3 -or $r[1].status -ne 'SELECTION_MISMATCH' -or $r[2].rows -ne 32768 -or $r[2].exit_code -ne 0) {throw 'Mismatch prevented 32K measurement'}
    if ((Get-Content (Join-Path $root 'dsa-window-32768.log') -Raw) -notmatch 'rows=32768') {throw '32K child did not run'}
    $defaultPlan=Get-Content (Join-Path $root 'dsa-window-plan.json') -Raw | ConvertFrom-Json
    if ($defaultPlan.identity -ne 'glm-pf-rtx-native-v1-dsa64-f32idx' -or $defaultPlan.runs[0].arguments -contains '--mma-index') {throw 'Default FP32 identity/arguments'}
    & powershell -NoProfile -ExecutionPolicy Bypass -File $launcher -BuildDir $root -Weights $weights -MmaIndex -PlanOnly | Set-Content (Join-Path $root 'plan.log')
    if ($LASTEXITCODE -ne 0) {throw 'MMA PlanOnly failed'}
    $plan=Get-Content (Join-Path $root 'dsa-window-plan.json') -Raw | ConvertFrom-Json
    if ($plan.identity -ne 'glm-pf-rtx-native-v1-dsa64-mmaidx' -or $plan.runs[0].arguments -notcontains '--mma-index' -or $plan.aggregate_timeout_seconds+$plan.kill_wait_seconds -gt 600) {throw 'MMA identity/argument/deadline contract'}
    & powershell -NoProfile -ExecutionPolicy Bypass -File $launcher -BuildDir $root -Weights $weights -VariantSweep | Set-Content (Join-Path $root 'sweep.log')
    if ($LASTEXITCODE -ne 3) {throw 'Sweep must preserve selection mismatch'}
    $sweep=Get-Content (Join-Path $root 'dsa-window-results.json') -Raw | ConvertFrom-Json
    if (@($sweep).Count -ne 11 -or @($sweep | Where-Object {$_.rows -eq 32768}).Count -ne 5) {throw 'Sweep did not run every 8K/32K variant'}
    if ($sweep[0].selection_measurement -ne 'RECORDED' -or $sweep[0].selection.differing_pool_memberships -ne 2 -or $sweep[0].device_ms -ne 12.5) {throw 'Sweep diagnostic counts/latency not recorded'}
    if ($sweep[0].reference_pairs.variant_vs_node_mimic.differing_pool_memberships -ne 2 -or $sweep[0].reference_pairs.node_mimic_vs_scalar_fp32.measurement -ne 'RECORDED') {throw 'Reference pair counts missing'}
    foreach ($v in @('fp32','mma','mma-proj-fp32-score','fp32-proj-mma-score','tf32x3')) {
        if (-not (Test-Path (Join-Path $root "dsa-window-$v-8192.log"))) {throw 'Variant log overwritten/missing'}
    }
    $env:STRIX_DSA_TEST_NO_REFERENCE='1'
    $ErrorActionPreference='Continue'
    & powershell -NoProfile -ExecutionPolicy Bypass -File $launcher -BuildDir $root -Weights $weights 2>&1 | Set-Content (Join-Path $root 'missing.log')
    $missingCode=$LASTEXITCODE;$ErrorActionPreference='Stop'
    if ($missingCode -eq 0) {throw 'Missing reference pair accepted'}
    $missing=Get-Content (Join-Path $root 'dsa-window-results.json') -Raw | ConvertFrom-Json
    if (@($missing).Count -ne 1 -or $missing[0].status -ne 'NO-REFERENCE') {throw 'Missing reference status/abort'}
    $env:STRIX_DSA_TEST_NO_REFERENCE=$null
    $env:STRIX_DSA_TEST_ROW='1024';$env:STRIX_DSA_TEST_CODE='1'
    $ErrorActionPreference='Continue'
    & powershell -NoProfile -ExecutionPolicy Bypass -File $launcher -BuildDir $root -Weights $weights 2>&1 | Set-Content (Join-Path $root 'error.log')
    $childCode=$LASTEXITCODE;$ErrorActionPreference='Stop'
    if ($childCode -eq 0) {throw 'Child error accepted'}
    $r=(Get-Content (Join-Path $root 'dsa-window-results.json') -Raw | ConvertFrom-Json)
    if (@($r).Count -ne 1 -or $r[0].status -ne 'ERROR') {throw 'Non-selection error must abort immediately'}
    Write-Output 'PASS DSA launcher: selection mismatch continues through 32K, aggregate failure preserved, MMA identity/plan/deadline, 11-case sweep with recorded counts, missing reference rejected, fatal child error aborts; no CUDA executed'
} finally {
    $env:CUDA_DEVICE_ORDER=$oldOrder;$env:CUDA_VISIBLE_DEVICES=$oldVisible
    $env:STRIX_DSA_TEST_ROW=$null;$env:STRIX_DSA_TEST_CODE=$null;$env:STRIX_DSA_TEST_NO_REFERENCE=$null
}
