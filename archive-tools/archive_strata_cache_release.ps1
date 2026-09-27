[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$fork = Join-Path $project 'research/strata-publication-20260927'
$candidate = Join-Path $project 'research/qwen-strata-update-20260927'
$stage = Join-Path $project 'research/strata-cache-release-20260927-v2'
$engineSha = '2e630a0b125a0b4f2be300678216d8aca2e086d1e5c2d9cb54855b55685c4c77'
if (Get-Process strata* -ErrorAction SilentlyContinue) { throw 'Stop packaging while an inference engine is running; do not terminate it automatically' }
if (Test-Path -LiteralPath $stage) { throw 'Staging already exists; preserve it' }
if (-not (Test-Path -LiteralPath (Join-Path $fork '.git'))) { throw 'Expected dedicated publication checkout' }
function Sha([string]$path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
if ((Sha (Join-Path $candidate 'build/strata.exe')) -ne $engineSha) { throw 'Frozen engine differs' }
$sourceManifestPath = Join-Path $fork 'local-evidence/source-snapshot-manifest.json'
$baselineManifest = Join-Path $fork 'local-evidence/source-snapshot-before-cache-20260927.json'
if (-not (Test-Path -LiteralPath $baselineManifest)) { Copy-Item -LiteralPath $sourceManifestPath -Destination $baselineManifest }
$sourceRows = [Collections.Generic.Dictionary[string,object]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($row in (Get-Content -LiteralPath $sourceManifestPath -Raw | ConvertFrom-Json).files) { $sourceRows[$row.path] = $row }
$textExtensions = @('.cpp','.hpp','.h','.c','.cu','.cuh','.py','.mjs','.sh','.ps1','.bat','.md','.cmake','.txt','.json','.patch')
function Check-PublicFile($file) {
    if ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Symlinks not admitted' }
    if ($file.Name -match '^(api[-.]?key|auth\.json|settings\.json|models\.json|requests.*\.jsonl|session.*\.jsonl)$') { throw 'Private artifact refused' }
}
function Copy-Source([string]$source, [string]$relative) {
    $file = Get-Item -LiteralPath $source
    Check-PublicFile $file
    if ($file.Length -gt 20MB) { throw "Oversized source: $relative" }
    $dest = Join-Path $fork $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
    Copy-Item -LiteralPath $source -Destination $dest
    $hash = Sha $source
    if ((Sha $dest) -ne $hash) { throw 'Copy checksum mismatch' }
    $sourceRows[$relative] = @{path=$relative;sha256=$hash;bytes=$file.Length}
}
function Source-Files([string]$directory) {
    foreach ($file in (Get-ChildItem -LiteralPath $directory)) {
        if ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Symlink in source tree' }
        if ($file.PSIsContainer) {
            if ($file.Name -notin @('.git','__pycache__','third_party','build','models','packs','.venv')) { Source-Files $file.FullName }
        } elseif ($file.Extension -in $textExtensions -or $file.Name -in @('.gitignore','.gitattributes','expert-profile.bin','draft_vocab.bin')) { $file }
    }
}
foreach ($file in (Get-ChildItem -LiteralPath $candidate -File)) {
    if ($file.Extension -in $textExtensions) { Copy-Source $file.FullName "deployment-snapshot/research/qwen-strata-update-20260927/$($file.Name)" }
}
foreach ($folder in @('source','manual')) {
    foreach ($file in (Source-Files (Join-Path $candidate $folder))) {
        $relative = $file.FullName.Substring($candidate.Length+1).Replace('\','/')
        Copy-Source $file.FullName "deployment-snapshot/research/qwen-strata-update-20260927/$relative"
        if ($folder -eq 'source') {
            $rootRelative = $relative.Substring(7)
            if ($rootRelative -notin @('README.md','.gitignore','.gitattributes')) { Copy-Source $file.FullName $rootRelative }
        }
    }
}
foreach ($name in @('run_qwen38_98k_ista_strata_server.bat','run_qwen38_98k_ista_strata_debug.bat','run_qwen38_98k_ista_strata_server.before-cache-20260927.bat')) {
    Copy-Source (Join-Path $project $name) "deployment-snapshot/$name"
}
foreach ($name in @('debug_launcher.py','test_debug_launcher.py','debug_launcher.before-cache-20260927.py')) {
    Copy-Source (Join-Path $project "research/qwen-strata-debug-20260927/$name") "deployment-snapshot/research/qwen-strata-debug-20260927/$name"
}
# The evidence projection never exports Pi prompts, traces, sessions or authentication.
$trialPath = Join-Path $candidate 'runs/update-r2-pi/timings.jsonl'
$proof = Get-Content -LiteralPath (Join-Path $candidate 'runs/update-r2-pi/pi-staged-vision-fixture02-result.json') -Raw | ConvertFrom-Json
$trialRows = @($proof.candidate_native_timings | ForEach-Object { $_.native_timings })
if (-not $proof.passed -or $proof.engine_sha256 -ne $engineSha -or $trialRows.Count -ne 2) { throw 'Unexpected existing Pi evidence' }
$numeric = @{schema='strata-cache-pi-numeric/v1';date='2026-09-27';engine_sha256=$engineSha;
    source_timings_sha256=(Sha $trialPath);source_attestation_sha256=(Sha (Join-Path $candidate 'runs/update-r2-pi/pi-staged-vision-fixture02-result.json'));
    rows=$trialRows;checks=$proof.checks;contains_prompts=$false;
    workload='Two successful finite agent requests, synthetic images/file, warm cache/retry; not a paired benchmark';
    user_confirmation=@{date='2026-09-27';successful_use=$true;pi_compact=$true;evidence_kind='user report, not instrumented benchmark';context_length_measured=$false}}
[IO.File]::WriteAllText((Join-Path $fork 'local-evidence/cache-pi-trial.json'),($numeric | ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
# Traverse only proof bindings; never scan the model directory or arbitrary user data.
$files = [Collections.Generic.Dictionary[string,object]]::new([StringComparer]::OrdinalIgnoreCase)
$pending = [Collections.Generic.Queue[string]]::new()
$visited = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$excluded = [Collections.Generic.List[object]]::new()
function Queue-Bound($object) {
    if ($null -eq $object) { return }
    if ($object -is [array]) { foreach ($value in $object) { Queue-Bound $value }; return }
    if ($object -is [pscustomobject]) {
        if ($object.PSObject.Properties.Name -contains 'path' -and $object.PSObject.Properties.Name -contains 'sha256') {
            $name = [string]$object.path
            if ([IO.Path]::IsPathRooted($name) -and (Test-Path -LiteralPath $name -PathType Leaf)) {
                $resolved = (Get-Item -LiteralPath $name).FullName
                if (-not $files.ContainsKey($resolved)) {
                    $files[$resolved] = $object.sha256
                    if ([IO.Path]::GetExtension($resolved) -eq '.json' -and $resolved.StartsWith($project+'\',[StringComparison]::OrdinalIgnoreCase) -and $resolved -notmatch '[\\/]qwen-strata-20260926[\\/](pack-iq3|mtp)[\\/]|[\\/]Project_ANTIREZ[\\/]models[\\/]') { $pending.Enqueue($resolved) }
                }
            }
        }
        foreach ($property in $object.PSObject.Properties) { Queue-Bound $property.Value }
    }
}
foreach ($lane in @('qwen-strata-20260926','qwen-strata-stock-20260926','qwen-strata-agent-20260927','qwen-strata-vision-20260927','qwen-strata-update-20260927')) {
    foreach ($file in (Get-ChildItem -LiteralPath (Join-Path $project "research/$lane") -Filter '*.json' -File)) { $pending.Enqueue($file.FullName) }
}
foreach ($lane in @('qwen-strata-agent-20260927','qwen-strata-update-20260927')) { $pending.Enqueue((Join-Path $project "research/$lane/manual/serve-only-gates.json")) }
while ($pending.Count) {
    $path = $pending.Dequeue()
    if ($visited.Add($path)) { Queue-Bound (Get-Content -LiteralPath $path -Raw | ConvertFrom-Json) }
}
foreach ($folder in @('qwen-strata-agent-20260927/build','qwen-strata-vision-20260927/build','qwen-strata-stock-20260926/official-engine','qwen-strata-update-20260927/build')) {
    foreach ($file in (Get-ChildItem -LiteralPath (Join-Path $project "research/$folder") -File | Where-Object Extension -in '.exe','.dll','.json')) { $files[$file.FullName] = Sha $file.FullName }
}
New-Item -ItemType Directory -Path (Join-Path $stage 'payload') | Out-Null
$runtimeRows = [Collections.Generic.List[object]]::new()
foreach ($entry in $files.GetEnumerator()) {
    $file = Get-Item -LiteralPath $entry.Key
    if (-not $file.FullName.StartsWith($project+'\',[StringComparison]::OrdinalIgnoreCase)) {
        $excluded.Add(@{name=$file.Name;reason='external dependency/model';sha256=$entry.Value;bytes=$file.Length}); continue
    }
    $relative = $file.FullName.Substring($project.Length+1).Replace('\','/')
    Check-PublicFile $file
    $exactEncoder = $relative -eq 'research/qwen-strata-stock-20260926/official-engine/strata-vision.exe' -and $file.Length -eq 90725376 -and $entry.Value -eq '98ffde98f5388b518d8581b2983973d785f20f00a21e5e82393b38f4188e094c'
    if ($relative -match '^(models/|research/qwen-strata-20260926/(pack-iq3|mtp)/)' -or $file.Extension -eq '.gguf' -or ($file.Length -gt 50MB -and -not $exactEncoder)) {
        $excluded.Add(@{name=$relative;reason='model-derived/external weight or oversized data';sha256=$entry.Value;bytes=$file.Length}); continue
    }
    if ($relative -match '/runs/' -and $file.Extension -eq '.jsonl' -and $file.Name -notin @('timings.jsonl','telemetry.jsonl')) { throw 'No conversation/trace JSONL in release' }
    $hash = Sha $file.FullName
    if ($hash -ne ([string]$entry.Value).ToLowerInvariant()) { throw "Bound artifact changed: $relative" }
    $destination = Join-Path $stage "payload/$relative"
    New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $destination
    if ((Sha $destination) -ne $hash) { throw 'Release copy mismatch' }
    $runtimeRows.Add(@{path=$relative;sha256=$hash;bytes=$file.Length})
}
$runtime = @{schema='strata-runtime-archive/v1';date='2026-09-27';engine_sha256=$engineSha;files=$runtimeRows;
    excluded_dependencies=$excluded;contains_model_weights=$false;clean_machine_restore_tested=$false}
$runtimeManifest = Join-Path $fork 'local-evidence/runtime-assets-manifest.json'
[IO.File]::WriteAllText($runtimeManifest,($runtime | ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
Copy-Item -LiteralPath $runtimeManifest -Destination (Join-Path $stage 'payload/runtime-assets-manifest.json')
$sourceManifest = @{schema='strata-source-archive/v1';date='2026-09-27';engine_sha256=$engineSha;files=@($sourceRows.Values | Sort-Object path)}
[IO.File]::WriteAllText($sourceManifestPath,($sourceManifest | ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
if (Get-Process strata* -ErrorAction SilentlyContinue) { throw 'Engine started; stop before compression and rerun packaging in a new staging location' }
Compress-Archive -LiteralPath (Join-Path $stage 'payload/research'),(Join-Path $stage 'payload/runtime-assets-manifest.json') -DestinationPath (Join-Path $stage 'windows-cache-runtime.zip') -CompressionLevel Optimal
$zipSha = Sha (Join-Path $stage 'windows-cache-runtime.zip')
[IO.File]::WriteAllText((Join-Path $stage 'windows-cache-runtime.zip.sha256'),($zipSha+'  windows-cache-runtime.zip'+[Environment]::NewLine),[Text.UTF8Encoding]::new($false))
if ((Sha (Join-Path $candidate 'build/strata.exe')) -ne $engineSha) { throw 'Active engine changed during publication' }
Write-Output "Source files: $($sourceRows.Count); runtime members: $($runtimeRows.Count); excluded dependencies: $($excluded.Count); ZIP SHA256: $zipSha. No model test or runtime change."
