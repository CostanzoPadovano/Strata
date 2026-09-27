$ErrorActionPreference='Stop'
$project=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$fork=Join-Path $project 'research/strata-publication-20260927'
$screens=@()
foreach($tag in @('dual-bench4k-six01','single-bench4k-six01','single-bench4k-six02','dual-bench4k-six02')){
    $probePath=Join-Path $project "research/qwen-strata-advanced-20260926/runs/$tag/probe.json"
    $resultPath=Join-Path $project "research/qwen-strata-advanced-20260926/runs/$tag/result.json"
    $probe=Get-Content -LiteralPath $probePath -Raw | ConvertFrom-Json
    $result=Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
    if(-not $probe.passed -or -not $result.passed){throw 'Screen did not pass'}
    $rows=@($probe.rows | Where-Object name -Match '^(decode|prefill)_screen_[0-9]+$' | ForEach-Object {
        @{name=$_.name;prompt_tokens=$_.native_timings.prompt_tokens;generated=$_.native_timings.generated;
          prompt_ms=$_.native_timings.prompt_ms;decode_ms=$_.native_timings.decode_ms;finish=$_.native_timings.finish}
    })
    if($rows.Count -ne 5){throw 'Unexpected screen shape'}
    $screens+=@{label=$tag;rows=$rows;passed=$true;source_probe_sha256=(Get-FileHash $probePath -Algorithm SHA256).Hash.ToLowerInvariant();
      source_result_sha256=(Get-FileHash $resultPath -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$evidence=@{schema='strata-numeric-4k/v1';context=4096;pool_workers=8;
    engine_sha256='f7d5bf01b658adec393e8d86a92d3da7026967b5179d910bfbebda46645d82d2';
    workload='3 x 42-prompt/256-output code; 2 x 3107-prompt/5-output table per screen';
    formula='arithmetic mean of per-request native token rates, pooled by variant';screens=$screens;contains_prompts=$false}
[IO.File]::WriteAllText((Join-Path $fork 'local-evidence/4k-screens.json'),($evidence|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
Write-Output 'Archived 20 numeric-only 4K throughput rows from four passing screens.'
