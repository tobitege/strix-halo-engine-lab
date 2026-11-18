# winevents.ps1: persistent Windows System-log watcher for gpu_guard.py (one PowerShell process, no per-second startup).
# Prints one line per poll:  HB <iso>                                   (heartbeat, every poll)
#                            EVT <iso> <provider> <id> <level> <text>    (each new matching event, oldest first)
#                            ERR <text>                               (fatal query failure, then exit)
# Matches: Microsoft-Windows-WHEA-Logger (any id: 17 = corrected PCIe error, 18/19/20 = fatal/corrected machine check),
#          nvlddmkm (any id: 13 graphics exception, 14 command re-execution, 153 TDR reset, ...), Display 4101 (TDR recovered).
# Read-only: never changes any system setting.
param([int]$IntervalMs = 1000, [string]$Since = "")
$ErrorActionPreference = 'Stop'
$start = if ($Since) { [datetime]::Parse($Since) } else { Get-Date }
$seen = @{}
$filter = @{ LogName = 'System'; ProviderName = @('Microsoft-Windows-WHEA-Logger', 'nvlddmkm', 'Display'); StartTime = $start }
while ($true) {
    $now = Get-Date
    try {
        $evs = Get-WinEvent -FilterHashtable $filter -ErrorAction Stop | Sort-Object RecordId
    } catch {
        # Get-WinEvent reports an empty matching set as an error; only that error is healthy.
        if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound,*') {
            $evs = @()
        } else {
            [Console]::Out.WriteLine(('ERR {0}' -f ($_.ToString() -replace '\s+', ' ')))
            [Console]::Out.Flush()
            exit 1
        }
    }
    foreach ($e in $evs) {
        if ($seen.ContainsKey($e.RecordId)) { continue }
        $seen[$e.RecordId] = 1
        if ($e.ProviderName -eq 'Display' -and $e.Id -ne 4101) { continue }
        $txt = ''
        if ($e.Message) { $txt = ($e.Message -replace '\s+', ' ') }
        if ($txt.Length -gt 200) { $txt = $txt.Substring(0, 200) }
        [Console]::Out.WriteLine(('EVT {0} {1} {2} {3} {4}' -f $e.TimeCreated.ToString('o'), $e.ProviderName, $e.Id, $e.LevelDisplayName, $txt))
    }
    # keep the window short: re-query from 30 s before now (events can be written late), dedup by RecordId
    $filter.StartTime = $now.AddSeconds(-30)
    if ($filter.StartTime -lt $start) { $filter.StartTime = $start }
    [Console]::Out.WriteLine(('HB {0}' -f (Get-Date).ToString('o')))
    [Console]::Out.Flush()
    Start-Sleep -Milliseconds $IntervalMs
}
