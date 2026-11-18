<#
.SYNOPSIS
Removes only unmodified files recorded by install.ps1 and empty owned directories.
Configuration, models, pre-existing content and modified installed files are kept.
#>
[CmdletBinding()]
param([string]$Prefix = (Join-Path $env:LOCALAPPDATA 'Programs\Strix'))
$ErrorActionPreference = 'Stop'
$Prefix = [IO.Path]::GetFullPath($Prefix).TrimEnd('\')
$manifestPath = Join-Path $Prefix 'install-manifest.json'
if (-not (Test-Path -LiteralPath $manifestPath)) {
    Write-Host "no ownership manifest in ${Prefix}: left untouched"
    exit 0
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$config = [IO.Path]::GetFullPath($manifest.ConfigDir).TrimEnd('\')
function Owned-Path([string]$relative) {
    $path = [IO.Path]::GetFullPath((Join-Path $Prefix $relative))
    if ($path -ne $Prefix -and -not $path.StartsWith($Prefix + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "manifest path escapes prefix: $relative"
    }
    for ($dir = $path; $dir; $dir = Split-Path $dir) {
        if ((Test-Path -LiteralPath $dir) -and
            ((Get-Item -LiteralPath $dir).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "manifest path traverses a reparse point: $relative"
        }
    }
    return $path
}
# Validate all paths before deleting anything. Never recurse through user content.
$files = @(foreach ($entry in $manifest.Files) {
    @{ Path = (Owned-Path $entry.Path); Sha256 = $entry.Sha256 }
})
$dirs = @(foreach ($dir in $manifest.Directories) { Owned-Path $dir })
if ($manifest.RegisteredTask -and (Get-ScheduledTask -TaskName 'Strix Server' -ErrorAction SilentlyContinue)) {
    Unregister-ScheduledTask -TaskName 'Strix Server' -Confirm:$false
    Write-Host "removed scheduled task 'Strix Server'"
}
$bin = Join-Path $Prefix 'bin'
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($userPath -and (($userPath -split ';') -contains $bin)) {
    [Environment]::SetEnvironmentVariable('Path', ((($userPath -split ';') | Where-Object { $_ -ne $bin }) -join ';'), 'User')
    Write-Host "removed $bin from the user PATH"
}
foreach ($entry in $files) {
    $path = $entry.Path
    if ($path -eq $config -or $path.StartsWith($config + '\', [StringComparison]::OrdinalIgnoreCase)) { continue }
    if ((Test-Path -LiteralPath $path -PathType Leaf) -and (Get-FileHash -LiteralPath $path).Hash -eq $entry.Sha256) {
        Remove-Item -LiteralPath $path -Force -Confirm:$false
    }
}
Remove-Item -LiteralPath $manifestPath -Force -Confirm:$false
foreach ($dir in ($dirs | Sort-Object Length -Descending | Select-Object -Unique)) {
    if ($dir -eq $config -or $dir.StartsWith($config + '\', [StringComparison]::OrdinalIgnoreCase)) { continue }
    if ((Test-Path -LiteralPath $dir -PathType Container) -and
        @(Get-ChildItem -LiteralPath $dir -Force).Count -eq 0) {
        Remove-Item -LiteralPath $dir -Force -Confirm:$false
    }
}
Write-Host "removed owned files from $Prefix; configuration and user content kept"
