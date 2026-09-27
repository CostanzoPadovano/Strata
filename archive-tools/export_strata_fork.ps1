[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$target = Join-Path $project 'research/strata-publication-20260927'
if (-not (Test-Path -LiteralPath (Join-Path $target '.git'))) { throw 'Expected isolated fork checkout' }
$manifest = [Collections.Generic.List[object]]::new()
$textExtensions = @('.cpp','.hpp','.h','.c','.cu','.cuh','.py','.mjs','.sh','.ps1','.bat','.md','.cmake','.txt','.json')
function Get-SourceFiles([string]$directory) {
    foreach ($entry in (Get-ChildItem -LiteralPath $directory)) {
        if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
        if ($entry.PSIsContainer) {
            if ($entry.Name -notin @('.git','__pycache__','third_party','build','build-native','pack','packs','models','.venv')) { Get-SourceFiles $entry.FullName }
        } else { $entry }
    }
}
function Copy-ArchivedFile([string]$source, [string]$relative) {
    $file = Get-Item -LiteralPath $source
    if ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Symlink not admitted' }
    if ($file.Length -gt 20MB) { throw "Unexpected large source: $relative" }
    if ($file.Name -match '^(api[-.]?key|auth\.json|settings\.json|models\.json|requests\.jsonl|session.*\.jsonl)$') { throw 'Private artifact refused' }
    $destination = Join-Path $target $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
    $before = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    $after = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($before -ne $after) { throw 'Copy identity mismatch' }
    $manifest.Add(@{path=$relative.Replace('\','/');sha256=$after;bytes=$file.Length})
}
$lanes = @('qwen-strata-20260926','qwen-strata-stock-20260926','qwen-strata-advanced-20260926','qwen-strata-agent-20260927','qwen-strata-vision-20260927','qwen-strata-debug-20260927')
foreach ($lane in $lanes) {
    $lanePath = Join-Path $project "research/$lane"
    # Top-level source/config/proof only; no runs, conversations, scratch, weights or build products.
    foreach ($file in (Get-ChildItem -LiteralPath $lanePath -File)) {
        if ($file.Extension -in $textExtensions) {
            Copy-ArchivedFile $file.FullName "deployment-snapshot/research/$lane/$($file.Name)"
        }
    }
    foreach ($subdir in @('source','manual','configs')) {
        $subPath = Join-Path $lanePath $subdir
        if (-not (Test-Path -LiteralPath $subPath -PathType Container)) { continue }
        # Avoid recursion into manual temporary directories and all Git metadata.
        $files = if ($subdir -eq 'manual' -or $subdir -eq 'configs') { Get-ChildItem -LiteralPath $subPath -File } else {
            Get-SourceFiles $subPath
        }
        foreach ($file in $files) {
            $relative = $file.FullName.Substring($lanePath.Length+1).Replace('\','/')
            if ($file.Extension -in $textExtensions -or $file.Name -in '.gitignore','.gitattributes' -or $relative -in 'source/data/expert-profile.bin','source/data/draft_vocab.bin') {
                Copy-ArchivedFile $file.FullName "deployment-snapshot/research/$lane/$relative"
                if ($lane -eq 'qwen-strata-vision-20260927' -and $subdir -eq 'source') {
                    $rootRelative = $relative.Substring(7)
                    if ($rootRelative -eq 'README.md') { $rootRelative = 'README-UPSTREAM.md' }
                    if ($rootRelative -notin '.gitattributes','.gitignore') { Copy-ArchivedFile $file.FullName $rootRelative }
                }
            }
        }
    }
}
foreach ($name in @('run_qwen38_98k_ista_strata_server.bat','run_qwen38_98k_ista_strata_debug.bat')) {
    Copy-ArchivedFile (Join-Path $project $name) "deployment-snapshot/$name"
}
Copy-ArchivedFile 'C:/MYPROJECT/TEST_QWEN/scripts/configure_pi_thinkingcap_wsl.mjs' 'deployment-snapshot/external/TEST_QWEN/scripts/configure_pi_thinkingcap_wsl.mjs'
# Publication evidence is numeric-only; neither request bodies nor session IDs are exported.
$timingPath = Join-Path $project 'research/qwen-strata-vision-20260927/runs/debug-20260927-113312-418800/timings.jsonl'
$rows = @(Get-Content -LiteralPath $timingPath | ForEach-Object { $_ | ConvertFrom-Json })
$safeRows = @($rows | Select-Object -Skip 4 | ForEach-Object {
    @{prompt_tokens=$_.native_timings.prompt_tokens;generated=$_.native_timings.generated;prompt_ms=$_.native_timings.prompt_ms;
      decode_ms=$_.native_timings.decode_ms;finish=$_.native_timings.finish;requested_max_new=$_.requested_max_new}
})
if ($safeRows.Count -ne 10) { throw 'Unexpected number of user trial timing rows' }
$evidence = @{schema='strata-numeric-trial/v1';lane='manual-pi-xhigh';date='2026-09-27';
    source_timings_sha256=(Get-FileHash -LiteralPath $timingPath -Algorithm SHA256).Hash.ToLowerInvariant();
    engine_sha256='f1072687e794aa5958f121c8967983947b5ea1322fa336c60688cfdc7736742f';
    capacity=98304;historical_output_cap=1024;rows=$safeRows;formula='sum(tokens)*1000/sum(milliseconds)';
    clean_exit_verified=$false;contains_prompts=$false}
$evidencePath = Join-Path $target 'local-evidence/pi-trial.json'
New-Item -ItemType Directory -Force -Path (Split-Path $evidencePath -Parent) | Out-Null
[IO.File]::WriteAllText($evidencePath,($evidence | ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
$manifestPath = Join-Path $target 'local-evidence/source-snapshot-manifest.json'
[IO.File]::WriteAllText($manifestPath,(@{schema='strata-source-archive/v1';date='2026-09-27';files=$manifest} | ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
Write-Output "Archived $($manifest.Count) byte-verified source/config files; 10 numeric-only timing rows. No model started."
