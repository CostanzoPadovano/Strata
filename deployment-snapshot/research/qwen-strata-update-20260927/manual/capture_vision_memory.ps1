[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$Run)
$ErrorActionPreference='Stop'
$visionMemoryRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../qwen-strata-vision-20260927/runs'))
$visionMemoryRun=[IO.Path]::GetFullPath($Run)
if(-not $visionMemoryRun.StartsWith($visionMemoryRoot+'\',[StringComparison]::OrdinalIgnoreCase)) {
    throw 'Snapshot outside the vision campaign'
}
$probe=Get-Content -LiteralPath (Join-Path $visionMemoryRun 'probe.json') -Raw | ConvertFrom-Json
if(-not $probe.api_ready -or -not $probe.vision_smoke_passed){throw 'Active checked image API required'}
$visionSnapshotKey=([IO.File]::ReadAllText('C:\llama_official\router\api-key.txt')).Trim()
$health=Invoke-RestMethod -Uri 'http://127.0.0.1:8037/health' -Headers @{Authorization="Bearer $visionSnapshotKey"} -TimeoutSec 10
if($health.runtime -ne 'ista-strata-vision-v1' -or $health.encoder_residency -ne 'on-demand'){throw 'Expected on-demand checked vision API'}
$entries=@(@{role='native';id=$probe.native_pid;name='strata'})
if($health.encoder_active){$entries+=@{role='encoder';id=$health.encoder_pid;name='strata-vision'}}
$rows=@()
foreach($entry in $entries) {
    $process=Get-Process -Id $entry.id
    if($process.ProcessName -ne $entry.name){throw 'Owned PID is gone or recycled'}
    $rows+=@{role=$entry.role;pid=$entry.id;working_set_bytes=$process.WorkingSet64;
        peak_working_set_bytes=$process.PeakWorkingSet64;private_bytes=$process.PrivateMemorySize64}
}
$snapshot=@{time_utc=[DateTime]::UtcNow.ToString('o');scope='Native/encoder physical working set, not Job/global commit';processes=$rows;encoder_active=$health.encoder_active}
$target=Join-Path $visionMemoryRun 'working-set-snapshot.json'
if(Test-Path -LiteralPath $target){throw 'Immutable snapshot already exists'}
[IO.File]::WriteAllText($target,($snapshot|ConvertTo-Json -Depth 6))
$snapshot|ConvertTo-Json -Depth 6
