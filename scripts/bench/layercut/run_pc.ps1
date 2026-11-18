param(
    [Parameter(Mandatory=$true)][string]$ModelDir,
    [Parameter(Mandatory=$true)][string]$Rope,
    [Parameter(Mandatory=$true)][string]$Prompt,
    [Parameter(Mandatory=$true)][string]$Node0,
    [Parameter(Mandatory=$true)][string]$Node1,
    [Parameter(Mandatory=$true)][int]$Port0,
    [Parameter(Mandatory=$true)][int]$Port1,
    [Parameter(Mandatory=$true)][ValidateRange(1,59)][int]$K,
    [int]$C,
    [int]$Repeat = 2,
    [ValidateSet('relay','direct')][string]$Act = 'relay',
    [ValidateSet('identity','bp')][string]$Codec = 'identity',
    [string]$PairIdentity = 'default/c4096',
    [string]$Binary = "$PSScriptRoot/../../../src/cuda/models/minimax/mm-rtx-prefill.exe",
    [string]$OutputDir = 'layercut-out'
)
$ErrorActionPreference = 'Stop'
if ($PSBoundParameters.ContainsKey('C') -and ($C -le 0 -or $C % 16 -ne 0)) { throw 'C must be a positive multiple of 16' }
if ($Repeat -lt 1) { throw 'Repeat must be positive' }
foreach ($port in @($Port0,$Port1)) { if ($port -lt 1 -or $port -gt 65535) { throw 'Invalid port' } }
if ((Get-Item -LiteralPath $Prompt).Length % 4 -ne 0) { throw 'Prompt must contain complete u32 token IDs' }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$arguments = @($ModelDir,$Rope,$OutputDir,"cut:$Prompt",'--layer-cut',"$K",
    '--cut-act',$Act,'--cut-pair-identity',$PairIdentity,'--resident',"$K",'--repeat',"$Repeat",
    '--send',"${Node0}:${Port0},${Node1}:${Port1}")
if ($PSBoundParameters.ContainsKey('C')) { $arguments += @('--cut-block',"$C") }
if ($Codec -eq 'bp') { $arguments += @('--codec','bp') }
& $Binary @arguments
if ($LASTEXITCODE -ne 0) { throw "Layer-cut producer failed: $LASTEXITCODE" }
