param([string]$Binary, [string]$Scratch)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $Scratch | Out-Null
$promptFile = Join-Path $Scratch 'prompt.u32'
[IO.File]::WriteAllBytes($promptFile, [byte[]]@(1,0,0,0))
$common = @{
    ModelDir = 'model'; Rope = 'rope.bin'; Prompt = $promptFile
    Node0 = 'node0.invalid'; Node1 = 'node1.invalid'; Port0 = 46000; Port1 = 46001
    K = 24; Binary = $Binary; OutputDir = (Join-Path $Scratch 'out')
}
$wrapper = Join-Path $PSScriptRoot '../scripts/bench/layercut/run_pc.ps1'
$result = & $wrapper @common
if (($result -join "`n") -notmatch 'split block 4096 \(identity grain\)') { throw 'Wrapper default block not derived from identity' }
$result = & $wrapper @common -PairIdentity 'default/c8192;test-policy'
if (($result -join "`n") -notmatch 'split block 8192 \(identity grain\)') { throw 'Custom identity not used' }
$result = & $wrapper @common -C 8192
if (($result -join "`n") -notmatch 'split block 8192 \(cli\)') { throw 'Explicit block not preserved' }
$result = & $wrapper @common -C 2048 -PairIdentity fp32
if (($result -join "`n") -notmatch 'split block 2048 \(cli\)') { throw 'FP32 explicit block rejected' }
$rejected = $false
try { & $wrapper @common -C 2048 | Out-Null } catch { $rejected = $true }
if (-not $rejected) { throw 'Identity-illegal explicit block accepted' }
Write-Output 'PASS: actual wrapper arguments and producer resolver'
