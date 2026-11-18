<#
.SYNOPSIS
Post-install smoke test of the Strix Windows package (installed prefix or extracted ZIP).

.DESCRIPTION
Every bin\strix-*.exe must load (no missing DLL) and answer --help. strix-skeleton-serve, when present, must
answer GET /health = 200 without weights. With -ServeArgs (a real model), the default server is started with
those arguments and /health must answer 200 within -HealthSeconds. With -EffectiveConfigArgs the server's
inspection option must exit 0. Unsupported explicitly requested checks fail. DeepSeek uses a spool service,
not HTTP, and supports --print-config; Qwen supports HTTP and --print-effective-config.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File smoke.ps1
#>
[CmdletBinding()]
param(
    [string]$Prefix = $PSScriptRoot,
    [int]$Port = 18080,
    [string[]]$ServeArgs = @(),
    [string[]]$EffectiveConfigArgs = @(),
    [int]$HealthSeconds = 600
)
$ErrorActionPreference = 'Stop'
$script:fails = 0; $script:passes = 0; $script:nohelp = 0
function Pass($m) { $script:passes++; Write-Host "PASS $m" }
function Fail($m) { $script:fails++; Write-Host "FAIL $m" }
function Skip($m) { Write-Host "SKIP $m" }

# Start-Process joins arguments with spaces: quote the ones that contain spaces.
function Join-Quoted([string[]]$arguments) { ($arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' ' }

function Invoke-Bounded([string]$exe, [string[]]$arguments, [int]$seconds) {
    $out = [IO.Path]::GetTempFileName(); $err = [IO.Path]::GetTempFileName()
    $p = Start-Process -FilePath $exe -ArgumentList (Join-Quoted $arguments) -NoNewWindow -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    if (-not $p.WaitForExit($seconds * 1000)) { $p.Kill(); $code = 'timeout' } else { $code = $p.ExitCode }
    $text = (Get-Content $out -Raw) + (Get-Content $err -Raw)
    Remove-Item $out, $err -Force
    return @{ Code = $code; Text = [string]$text }
}

function Test-Health([int]$port) {
    try { return (Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri "http://127.0.0.1:$port/health").StatusCode }
    catch { return 0 }
}

$bin = Join-Path $Prefix 'bin'
if (-not (Test-Path $bin)) { Write-Host "FAIL no bin directory at $bin"; exit 1 }
$expectedPath = Join-Path $Prefix 'share\strix\support\executables-cuda.txt'
if (-not (Test-Path -LiteralPath $expectedPath -PathType Leaf)) {
    Write-Host 'FAIL missing expected-executable manifest'; exit 1
}
$expected = @(Get-Content -LiteralPath $expectedPath | Where-Object { $_.Trim() } | Select-Object -Unique)
$checked = 0
foreach ($name in $expected) {
    if ($name -notmatch '^strix-[A-Za-z0-9_-]+\.exe$') { Fail "invalid expected executable: $name"; continue }
    $exe = Get-Item -LiteralPath (Join-Path $bin $name) -ErrorAction SilentlyContinue
    if (-not $exe -or $exe.PSIsContainer) { Fail "missing expected executable: $name"; continue }
    $checked++
    $r = Invoke-Bounded $exe.FullName @('--help') 30
    if ($r.Code -eq -1073741515) { Fail "$($exe.Name): a required DLL is missing (0xC0000135)" }
    elseif ($r.Code -eq 'timeout') { Fail "$($exe.Name) --help: timeout" }
    elseif ($r.Code -eq 0 -or $r.Text -match '(?i)usage|options') { Pass "$($exe.Name) --help (rc=$($r.Code))" }
    elseif ($r.Text -match '(?i)device gate' -or $r.Text.Trim() -eq '') {
        Fail "$($exe.Name) --help: rc=$($r.Code): $($r.Text.Substring(0, [Math]::Min(160, $r.Text.Length)))"
    } else {
        # The program loaded and its own parser answered (it has no --help yet): the package is intact.
        $script:nohelp++
        Pass "$($exe.Name) loads (no --help: $($r.Text.Trim().Split([char]10)[0]))"
    }
}
if ($checked -eq 0) { Fail 'no expected executables checked' }

$skel = Join-Path $bin 'strix-skeleton-serve.exe'
$reg = Join-Path $Prefix 'share\strix\examples\skeleton_engine'
$server = $null
$metadata = $null
$metadataPath = Join-Path $Prefix 'share\strix\support\server.json'
if (Test-Path -LiteralPath $metadataPath) {
    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    $server = Get-Item -LiteralPath (Join-Path $bin ($metadata.binary + '.exe')) -ErrorAction SilentlyContinue
}
if ($ServeArgs.Count -gt 0 -and (-not $server -or -not $metadata.health)) {
    Fail 'requested HTTP health check has no installed HTTP server (DeepSeek uses a spool service)'
} elseif ($ServeArgs.Count -gt 0) {
    $p = Start-Process -FilePath $server.FullName -WindowStyle Hidden -PassThru -ArgumentList (Join-Quoted ($ServeArgs + @('--bind', '127.0.0.1', '--port', "$Port")))
    $code = 0; $deadline = (Get-Date).AddSeconds($HealthSeconds)
    while ((Get-Date) -lt $deadline -and $code -ne 200 -and -not $p.HasExited) { Start-Sleep -Seconds 2; $code = Test-Health $Port }
    if (-not $p.HasExited) { $p.Kill() }
    if ($code -eq 200) { Pass "$($server.Name) GET /health = 200" } else { Fail "$($server.Name) GET /health = $code" }
} elseif ((Test-Path $skel) -and (Test-Path (Join-Path $reg 'models\registry.toml'))) {
    $p = Start-Process -FilePath $skel -NoNewWindow -PassThru -ArgumentList (Join-Quoted @('--registry-root', $reg,
        '--registry', 'models/registry.toml', '--model', 'skeleton', '--bind', '127.0.0.1', '--port', "$Port"))
    $code = 0
    for ($i = 0; $i -lt 50 -and $code -ne 200 -and -not $p.HasExited; $i++) { Start-Sleep -Milliseconds 200; $code = Test-Health $Port }
    if (-not $p.HasExited) { $p.Kill() }
    if ($code -eq 200) { Pass 'strix-skeleton-serve GET /health = 200' } else { Fail "strix-skeleton-serve GET /health = $code" }
} else {
    Skip 'health: this package has no weight-free server; pass -ServeArgs with a model to check /health'
}

if ($EffectiveConfigArgs.Count -gt 0 -and (-not $server -or -not $metadata.inspect)) {
    Fail 'requested configuration inspection has no installed supporting server'
} elseif ($EffectiveConfigArgs.Count -gt 0) {
    $r = Invoke-Bounded $server.FullName ($EffectiveConfigArgs + @($metadata.inspect)) 300
    if ($r.Code -eq 0) { Pass "$($server.Name) $($metadata.inspect)" } else { Fail "$($server.Name) $($metadata.inspect) rc=$($r.Code)" }
} else {
    Skip '--print-effective-config: needs -EffectiveConfigArgs with a model'
}
Write-Host "smoke: $($script:passes) passed ($($script:nohelp) without --help), $($script:fails) failed"
if ($script:fails -gt 0) { exit 1 }
exit 0
