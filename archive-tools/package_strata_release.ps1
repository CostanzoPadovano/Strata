[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$project=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$fork=Join-Path $project 'research/strata-publication-20260927'
$stage=Join-Path $project 'research/strata-publication-assets-20260927-v2'
if(Test-Path -LiteralPath $stage){throw 'Release staging exists; preserve it, do not overwrite'}
New-Item -ItemType Directory -Path (Join-Path $stage 'payload') | Out-Null
$files=[Collections.Generic.Dictionary[string,object]]::new([StringComparer]::OrdinalIgnoreCase)
$pending=[Collections.Generic.Queue[string]]::new()
$visited=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$excluded=[Collections.Generic.List[object]]::new()
function Queue-Bound($object) {
    if($null -eq $object){return}
    if($object -is [array]){foreach($value in $object){Queue-Bound $value};return}
    if($object -is [pscustomobject]){
        if($object.PSObject.Properties.Name -contains 'path' -and $object.PSObject.Properties.Name -contains 'sha256'){
            $name=[string]$object.path
            if([IO.Path]::IsPathRooted($name) -and (Test-Path -LiteralPath $name -PathType Leaf)){
                $resolved=(Get-Item -LiteralPath $name).FullName
                if(-not $files.ContainsKey($resolved)){
                    $files[$resolved]=$object.sha256
                    if([IO.Path]::GetExtension($resolved) -eq '.json' -and $resolved.StartsWith($project+'\',[StringComparison]::OrdinalIgnoreCase) -and $resolved -notmatch '[\\/]pack-iq3[\\/]|[\\/]mtp[\\/]|[\\/]models[\\/]'){$pending.Enqueue($resolved)}
                }
            }
        }
        foreach($property in $object.PSObject.Properties){Queue-Bound $property.Value}
    }
}
foreach($lane in @('qwen-strata-20260926','qwen-strata-stock-20260926','qwen-strata-agent-20260927','qwen-strata-vision-20260927')){
    foreach($file in (Get-ChildItem -LiteralPath (Join-Path $project "research/$lane") -Filter '*.json' -File)){$pending.Enqueue($file.FullName)}
}
$pending.Enqueue((Join-Path $project 'research/qwen-strata-agent-20260927/manual/serve-only-gates.json'))
while($pending.Count){
    $path=$pending.Dequeue()
    if($visited.Add($path)){Queue-Bound (Get-Content -LiteralPath $path -Raw | ConvertFrom-Json)}
}
foreach($folder in @('qwen-strata-agent-20260927/build','qwen-strata-vision-20260927/build','qwen-strata-stock-20260926/official-engine')){
    foreach($file in (Get-ChildItem -LiteralPath (Join-Path $project "research/$folder") -File | Where-Object Extension -in '.exe','.dll','.json')){
        $files[$file.FullName]=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$rows=[Collections.Generic.List[object]]::new()
foreach($entry in $files.GetEnumerator()){
    $file=Get-Item -LiteralPath $entry.Key
    if(-not $file.FullName.StartsWith($project+'\',[StringComparison]::OrdinalIgnoreCase)){
        $excluded.Add(@{name=$file.Name;reason='external dependency/model';sha256=$entry.Value;bytes=$file.Length});continue
    }
    $relative=$file.FullName.Substring($project.Length+1).Replace('\','/')
    if($file.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'No symlinks'}
    if($relative -match '/pack-iq3/|/mtp/|/models/' -or $file.Extension -eq '.gguf' -or $file.Length -gt 50MB){
        $excluded.Add(@{name=$relative;reason='model-derived/external weight or oversized data';sha256=$entry.Value;bytes=$file.Length});continue
    }
    if($file.Name -match '^(api[-.]?key|auth\.json|settings\.json|models\.json|requests\.jsonl|session.*\.jsonl)$'){throw 'Credential/user request refused'}
    $hash=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if($hash -ne ([string]$entry.Value).ToLowerInvariant()){throw "Bound artifact changed: $relative"}
    $destination=Join-Path $stage "payload/$relative"
    New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $destination
    if((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hash){throw 'Release copy mismatch'}
    $rows.Add(@{path=$relative;sha256=$hash;bytes=$file.Length})
}
$result=@{schema='strata-runtime-archive/v1';date='2026-09-27';files=$rows;excluded_dependencies=$excluded;contains_model_weights=$false;clean_machine_restore_tested=$false}
$manifest=Join-Path $fork 'local-evidence/runtime-assets-manifest.json'
[IO.File]::WriteAllText($manifest,($result|ConvertTo-Json -Depth 7),[Text.UTF8Encoding]::new($false))
Copy-Item -LiteralPath $manifest -Destination (Join-Path $stage 'payload/runtime-assets-manifest.json')
Compress-Archive -LiteralPath (Join-Path $stage 'payload/research'),(Join-Path $stage 'payload/runtime-assets-manifest.json') -DestinationPath (Join-Path $stage 'windows-runtime.zip') -CompressionLevel Optimal
$zipHash=(Get-FileHash -LiteralPath (Join-Path $stage 'windows-runtime.zip') -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $stage 'windows-runtime.zip.sha256'),($zipHash+'  windows-runtime.zip'+[Environment]::NewLine),[Text.UTF8Encoding]::new($false))
Write-Output "Release: $($rows.Count) exact files, $($excluded.Count) explicitly excluded dependencies; SHA256 $zipHash"
