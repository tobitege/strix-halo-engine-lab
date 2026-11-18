param(
    [Parameter(Mandatory=$true)][string]$BuildDir,
    [string]$Weights = '',
    [string]$Histogram = '',
    [ValidateSet('legacy','cutlass','both')][string]$Path = 'cutlass'
)
# This launcher is an explicit GPU run. A single 570-second aggregate deadline
# covers synthetic routed/shared/dense and (when supplied) real routed tensors.
$ErrorActionPreference = 'Stop'
if ($env:CUDA_DEVICE_ORDER -ne 'PCI_BUS_ID' -or $env:CUDA_VISIBLE_DEVICES -notmatch '^GPU-[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$') {
    throw 'Set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES to exactly one PRO GPU UUID'
}
$build = (Resolve-Path $BuildDir).Path
$deadline = [DateTime]::UtcNow.AddSeconds(570)
$runs = @(@('--kind','routed'), @('--kind','shared'), @('--kind','dense'))
if ($Weights) {
    $Weights = (Resolve-Path $Weights).Path
    $manifest = Get-Content ($Weights + '.json') -Raw | ConvertFrom-Json
    if ((Get-FileHash $Weights -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.sha256) {
        throw 'Real archive SHA256 differs from export manifest'
    }
    $real = @('--kind','routed','--weights', ('"' + $Weights + '"'))
    if ($Histogram) { $real += @('--histogram', ('"' + (Resolve-Path $Histogram).Path + '"')) }
    $runs += ,$real
} else { Write-Output 'NO-REFERENCE: real EXL3 archive not supplied' }
$arms = if ($Path -eq 'both') { @('legacy','cutlass') } else { @($Path) }
$suite = @()
foreach ($run in $runs) { foreach ($arm in $arms) { $suite += ,(@('--path',$arm) + $run) } }
$runs = $suite
for ($j=0; $j -lt $runs.Count; $j++) {
    $left = [int][Math]::Floor(($deadline - [DateTime]::UtcNow).TotalMilliseconds)
    if ($left -le 0) { throw 'Window exhausted; suite incomplete' }
    $stdout = Join-Path $build "window-$j.log"
    $stderr = Join-Path $build "window-$j.err"
    $arguments = @('--run','--iters','3') + $runs[$j]
    $p = Start-Process -FilePath (Join-Path $build 'glm_native_bench.exe') -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if (-not $p.WaitForExit($left)) { $p.Kill(); $p.WaitForExit(); throw '570-second deadline; partial run is not a pass' }
    $p.Refresh()
    if ($p.ExitCode -ne 0) { throw "Benchmark failed; inspect $stderr" }
    Get-Content $stdout
}
