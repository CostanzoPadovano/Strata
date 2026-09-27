[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$Tag,[ValidateRange(10,120)][int]$Seconds=60,[string]$ObserveRun)
$ErrorActionPreference='Stop'
if($Tag -notmatch '^[a-zA-Z0-9-]+$'){throw 'Invalid evidence tag'}
$visionBaselineRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../qwen-strata-vision-20260927/runs'))
$visionBaselineRun=Join-Path $visionBaselineRoot $Tag
if(Test-Path -LiteralPath $visionBaselineRun){throw 'Immutable evidence already exists'}
New-Item -ItemType Directory -Path $visionBaselineRun | Out-Null
$target=Join-Path $visionBaselineRun 'baseline.jsonl'
$timer=[Diagnostics.Stopwatch]::StartNew()
$nextReport=0
while($timer.Elapsed.TotalSeconds -lt $Seconds){
    $memory=Get-CimInstance Win32_PerfRawData_PerfOS_Memory
    $processes=@(Get-Process -ErrorAction SilentlyContinue)
    $privateTotal=($processes|Measure-Object PrivateMemorySize64 -Sum).Sum
    $wsTotal=($processes|Measure-Object WorkingSet64 -Sum).Sum
    $largest=@($processes|Sort-Object PrivateMemorySize64 -Descending|Select-Object -First 8 ProcessName,Id,PrivateMemorySize64,WorkingSet64)
    $vm=@($processes|Where-Object {$_.ProcessName -eq 'vmmemWSL'}|Select-Object ProcessName,Id,PrivateMemorySize64,WorkingSet64)
    $gitProcesses=@($processes|Where-Object {$_.ProcessName -eq 'git'})
    $gitPrivate=($gitProcesses|Measure-Object PrivateMemorySize64 -Sum).Sum
    $gitWorkingSet=($gitProcesses|Measure-Object WorkingSet64 -Sum).Sum
    $gpuMemory=@(Get-CimInstance Win32_PerfRawData_GPUPerformanceCounters_GPUProcessMemory|Sort-Object TotalCommitted -Descending|Select-Object -First 12 Name,TotalCommitted,DedicatedUsage,SharedUsage)
    $phase=$null
    if($ObserveRun){
        $observed=[IO.Path]::GetFullPath((Join-Path $visionBaselineRoot $ObserveRun))
        if(-not $observed.StartsWith($visionBaselineRoot+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Observed run outside campaign'}
        $phaseFile=Join-Path $observed 'phase.json'
        if(Test-Path -LiteralPath $phaseFile){try{$phase=Get-Content -LiteralPath $phaseFile -Raw|ConvertFrom-Json}catch{}}
    }
    $row=@{time_utc=[DateTime]::UtcNow.ToString('o');elapsed_seconds=$timer.Elapsed.TotalSeconds;
        commit_bytes=[long]$memory.CommittedBytes;commit_limit_bytes=[long]$memory.CommitLimit;
        available_bytes=[long]$memory.AvailableBytes;pool_paged_bytes=[long]$memory.PoolPagedBytes;
        pool_nonpaged_bytes=[long]$memory.PoolNonpagedBytes;all_process_private_bytes=[long]$privateTotal;
        all_process_working_set_bytes=[long]$wsTotal;private_accounting_residual_bytes=([long]$memory.CommittedBytes-[long]$privateTotal);
        process_count=$processes.Count;largest_private=$largest;wsl_vm=$vm;gpu_process_memory=$gpuMemory;observed_phase=$phase;
        git_process_count=$gitProcesses.Count;git_private_bytes=[long]$gitPrivate;git_working_set_bytes=[long]$gitWorkingSet;
        scope='Read-only idle baseline; sums are not unique physical RAM; inaccessible/shared/kernel accounting may be missing'}
    [IO.File]::AppendAllText($target,($row|ConvertTo-Json -Depth 5 -Compress)+[Environment]::NewLine)
    if($timer.Elapsed.TotalSeconds -ge $nextReport){
        @{elapsed=[math]::Round($timer.Elapsed.TotalSeconds,1);commit_free_gib=([math]::Round(($memory.CommitLimit-$memory.CommittedBytes)/1GB,2));ram_free_gib=([math]::Round($memory.AvailableBytes/1GB,2))}|ConvertTo-Json -Compress
        $nextReport+=10
    }
    Start-Sleep -Seconds 2
}
"Idle baseline saved: $target"
