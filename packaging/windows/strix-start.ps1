<#
.SYNOPSIS
Starts the Strix server with the arguments in <ConfigDir>\serve.args (default %ProgramData%\Strix).

.DESCRIPTION
Runs in the foreground (the "Strix Server" scheduled task or a console) and appends the server output to
<ConfigDir>\logs\strix-serve.log. When the chosen server accepts --serve-config (it lists it in --help),
<ConfigDir>\serve.toml is passed with it; options on serve.args lines win over the file.
#>
[CmdletBinding()]
param(
    [string]$ConfigDir = (Join-Path $env:ProgramData 'Strix'),
    [string]$Binary = '',
    [switch]$Console
)
$ErrorActionPreference = 'Stop'
$bin = Join-Path $PSScriptRoot 'bin'
$argsFile = Join-Path $ConfigDir 'serve.args'
$argv = New-Object System.Collections.Generic.List[string]
if (Test-Path $argsFile) {
    foreach ($raw in Get-Content $argsFile) {
        $line = $raw.Trim()
        if ($line -eq '' -or $line.StartsWith('#')) { continue }
        if ($line -match '^binary\s+(\S+)$') { if ($Binary -eq '') { $Binary = $Matches[1] }; continue }
        $space = $line.IndexOf(' ')
        if ($space -lt 0) { $argv.Add($line) }
        else { $argv.Add($line.Substring(0, $space)); $argv.Add($line.Substring($space + 1).Trim()) }
    }
}
if ($Binary -eq '') {
    $metadata = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'share\strix\support\server.json') -Raw | ConvertFrom-Json
    $Binary = $metadata.binary
}
$exe = Join-Path $bin "$Binary.exe"
if (-not (Test-Path $exe)) { throw "strix-start.ps1: no server binary '$Binary' in $bin" }

$devices = Join-Path $ConfigDir 'devices.toml'
if ((Test-Path $devices) -and -not ($argv.Contains('--devices') -or $argv.Contains('--device-id'))) {
    $argv.Insert(0, $devices)
    $argv.Insert(0, '--devices')
}
$toml = Join-Path $ConfigDir 'serve.toml'
if ((Test-Path $toml) -and ((& $exe --help 2>&1 | Out-String) -match '--serve-config')) {
    $argv.Insert(0, $toml)
    $argv.Insert(0, '--serve-config')
}

if ($Console) {
    & $exe @argv
    exit $LASTEXITCODE
}
$logDir = Join-Path $ConfigDir 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'strix-serve.log'
Add-Content -Path $log -Value ("--- " + (Get-Date -Format s) + " start " + $Binary + " " + ($argv -join ' '))
& $exe @argv *>> $log
exit $LASTEXITCODE
