param(
    [Parameter(Mandatory=$true)][string]$Executable,
    [Parameter(Mandatory=$true)][string]$OutputDir
)
# Short, device-0-only operator measurements. Run after the build has exited.
$ErrorActionPreference = 'Stop'
$env:CUDA_VISIBLE_DEVICES = '0'
$exe = (Resolve-Path -LiteralPath $Executable).Path
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$out = (Resolve-Path -LiteralPath $OutputDir).Path
$smi = (Get-Command nvidia-smi).Source
& $smi -i 0 --query-gpu=name,driver_version,compute_cap,memory.total --format=csv |
    Set-Content -LiteralPath (Join-Path $out 'device.csv')
if ($LASTEXITCODE) { throw 'device inventory failed' }
foreach ($shape in @(@(5120,5120), @(17408,5120), @(5120,17408))) {
    foreach ($m in @(8192,1,2,3,4,5,6,7,8)) {
        $n = $shape[0]; $k = $shape[1]
        $stem = "m${m}-n${n}-k${k}"
        $stdout = Join-Path $out "$stem.txt"
        $stderr = Join-Path $out "$stem-errors.txt"
        $process = Start-Process -FilePath $exe -ArgumentList @('--bench',$m,$n,$k) `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
            -WindowStyle Hidden -PassThru
        try {
            if (!$process.WaitForExit(110000)) {
                $process.Kill(); $process.WaitForExit()
                throw "benchmark exceeded 110 seconds: $stem"
            }
            if ($process.ExitCode) { throw "benchmark failed: $stem (see $stderr)" }
            Get-Content -LiteralPath $stdout
        } finally { $process.Dispose() }
    }
}
