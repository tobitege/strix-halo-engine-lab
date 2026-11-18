<#
.SYNOPSIS
Installs the Strix CUDA package from this extracted ZIP.

.DESCRIPTION
Copies bin\ and share\ to -Prefix (default: per-user, no administrator rights needed), creates the
configuration directory (default %ProgramData%\Strix) with serve.toml and serve.args unless they already exist
(an existing configuration is never overwritten), and optionally registers a "Strix Server" scheduled task that
starts strix-start.ps1 at logon of the installing user. Nothing is downloaded; no model is installed.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\install.ps1 -AddToPath -RegisterTask
#>
[CmdletBinding()]
param(
    [string]$Prefix = (Join-Path $env:LOCALAPPDATA 'Programs\Strix'),
    [string]$ConfigDir = (Join-Path $env:ProgramData 'Strix'),
    [switch]$AddToPath,
    [switch]$RegisterTask
)
$ErrorActionPreference = 'Stop'
$src = $PSScriptRoot
if (-not (Test-Path (Join-Path $src 'bin'))) { throw "install.ps1: run it from the extracted package (no bin\ next to it)" }

$Prefix = [IO.Path]::GetFullPath($Prefix).TrimEnd('\')
$ConfigDir = [IO.Path]::GetFullPath($ConfigDir).TrimEnd('\')
# Validate directory destinations and their ancestors before copying any payload.
foreach ($destination in @($Prefix, $ConfigDir)) {
    for ($dir = $destination; $dir; $dir = Split-Path $dir) {
        if (Test-Path -LiteralPath $dir) {
            $item = Get-Item -LiteralPath $dir
            if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                throw "destination must traverse ordinary directories: $dir"
            }
        }
    }
}
foreach ($name in 'serve.toml', 'serve.args', 'devices.example.toml') {
    $dst = Join-Path $ConfigDir $name
    if ((Test-Path -LiteralPath $dst) -and -not (Test-Path -LiteralPath $dst -PathType Leaf)) {
        throw "configuration destination is not a file: $dst"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $src "etc\strix\$name") -PathType Leaf)) {
        throw "missing configuration template: $name"
    }
}
$manifestPath = Join-Path $Prefix 'install-manifest.json'
$previous = @{}
$old = $null
if (Test-Path -LiteralPath $manifestPath) {
    $old = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    foreach ($entry in $old.Files) { $previous[$entry.Path] = $entry.Sha256 }
}
$files = @(foreach ($dir in 'bin', 'share') {
    Get-ChildItem -LiteralPath (Join-Path $src $dir) -File -Recurse -ErrorAction SilentlyContinue
}) + @(Get-ChildItem -Path (Join-Path $src '*.ps1') -File)
# Preflight all destinations; do not adopt or overwrite unrelated files.
foreach ($file in $files) {
    $relative = $file.FullName.Substring($src.Length + 1)
    $dst = Join-Path $Prefix $relative
    if ($dst -eq $ConfigDir -or $dst.StartsWith($ConfigDir + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "configuration overlaps an installed file: $dst"
    }
    for ($dir = $dst; $dir; $dir = Split-Path $dir) {
        if ((Test-Path -LiteralPath $dir) -and
            ((Get-Item -LiteralPath $dir).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "installation path traverses a reparse point: $dir"
        }
    }
    if ((Test-Path -LiteralPath $dst) -and
        (-not $previous.ContainsKey($relative) -or
         (Get-FileHash -LiteralPath $dst).Hash -ne $previous[$relative])) {
        throw "refusing to overwrite unowned or modified file: $dst"
    }
}
if ((Test-Path -LiteralPath (Join-Path $Prefix 'install.txt')) -and -not $previous.ContainsKey('install.txt')) {
    throw 'existing install.txt has no ownership manifest; leave the old installation intact'
}
$ownedDirs = @()
if ($old) { $ownedDirs += @($old.Directories) }
# Back up overwritten files and record new files before each write, including a
# possibly partial copy. The old ownership manifest stays intact until commit.
$createdPrefix = -not (Test-Path -LiteralPath $Prefix)
New-Item -ItemType Directory -Force -Path $Prefix | Out-Null
if ($createdPrefix) { $ownedDirs += '.' }
$transaction = Join-Path $Prefix ('.strix-install-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $transaction | Out-Null
$undo = [Collections.Generic.List[object]]::new()
function Save-InstallFile([string]$path) {
    $backup = $null
    if (Test-Path -LiteralPath $path) {
        $backup = Join-Path $transaction ([string]$undo.Count)
        [IO.File]::Copy($path, $backup)
    }
    $undo.Add(@{ Path = $path; Backup = $backup })
}
try {
foreach ($file in $files) {
    $relative = $file.FullName.Substring($src.Length + 1)
    $dst = Join-Path $Prefix $relative
    for ($dir = Split-Path $dst; $dir -and -not (Test-Path -LiteralPath $dir); $dir = Split-Path $dir) {
        if ($dir -eq $Prefix) { $ownedDirs += '.'; break }
        $ownedDirs += $dir.Substring($Prefix.Length + 1)
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
    Save-InstallFile $dst
    Copy-Item -LiteralPath $file.FullName -Destination $dst -Force
    $previous[$relative] = (Get-FileHash -LiteralPath $dst).Hash
}

New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null
foreach ($name in 'serve.toml', 'serve.args', 'devices.example.toml') {
    $dst = Join-Path $ConfigDir $name
    if (Test-Path $dst) { Write-Host "kept existing $dst" }
    else { Save-InstallFile $dst; Copy-Item -Path (Join-Path $src "etc\strix\$name") -Destination $dst; Write-Host "wrote $dst" }
}

$receipt = @("prefix: $Prefix", "config: $ConfigDir")
$build = Get-ChildItem -Path (Join-Path $Prefix 'share\strix\support') -Filter 'build-*.txt' -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($build) { $receipt += Get-Content $build.FullName -TotalCount 4 }
Save-InstallFile (Join-Path $Prefix 'install.txt')
Set-Content -Path (Join-Path $Prefix 'install.txt') -Value $receipt
$previous['install.txt'] = (Get-FileHash -LiteralPath (Join-Path $Prefix 'install.txt')).Hash
$manifest = @{
    ConfigDir = $ConfigDir
    RegisteredTask = [bool]($RegisterTask -or ($old -and $old.RegisteredTask))
    Directories = @($ownedDirs | Select-Object -Unique)
    Files = @(foreach ($name in $previous.Keys) { @{ Path = $name; Sha256 = $previous[$name] } })
}
$pendingManifest = Join-Path $transaction 'manifest.json'
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $pendingManifest
if (Test-Path -LiteralPath $manifestPath) {
    [IO.File]::Replace($pendingManifest, $manifestPath, (Join-Path $transaction 'previous-manifest.json'))
} else {
    [IO.File]::Move($pendingManifest, $manifestPath)
}
} catch {
    for ($i = $undo.Count - 1; $i -ge 0; $i--) {
        $entry = $undo[$i]
        if ($entry.Backup) { [IO.File]::Copy($entry.Backup, $entry.Path, $true) }
        elseif (Test-Path -LiteralPath $entry.Path) { Remove-Item -LiteralPath $entry.Path -Force }
    }
    throw
} finally {
    Get-ChildItem -LiteralPath $transaction -File | Remove-Item -Force
    Remove-Item -LiteralPath $transaction
}

# Optional OS integration runs only after payload ownership is committed. A
# failure here leaves a valid, owned installation that can be retried safely.
if ($AddToPath) {
    $bin = Join-Path $Prefix 'bin'
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not (($userPath -split ';') -contains $bin)) {
        [Environment]::SetEnvironmentVariable('Path', (($userPath.TrimEnd(';') + ';' + $bin).TrimStart(';')), 'User')
        Write-Host "added $bin to the user PATH (new shells)"
    }
}
if ($RegisterTask) {
    $start = Join-Path $Prefix 'strix-start.ps1'
    $arg = '-NoProfile -ExecutionPolicy Bypass -File "' + $start + '" -ConfigDir "' + $ConfigDir + '"'
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arg -WorkingDirectory $Prefix
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries
    Register-ScheduledTask -TaskName 'Strix Server' -Action $action -Trigger $trigger -Settings $settings `
        -Description 'Strix inference server (strix-start.ps1)' -Force | Out-Null
    Write-Host "registered scheduled task 'Strix Server' (at logon of $env:USERNAME; not started now)"
}
Write-Host "installed to $Prefix; edit $ConfigDir\serve.args (model paths), then run strix-start.ps1"
Write-Host "smoke test: powershell -ExecutionPolicy Bypass -File `"$Prefix\smoke.ps1`""
