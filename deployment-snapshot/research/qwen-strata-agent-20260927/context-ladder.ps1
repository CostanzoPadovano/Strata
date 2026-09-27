param([string]$Suffix = '01')
$ErrorActionPreference = 'Stop'
if ($Suffix -notmatch '^[a-zA-Z0-9_-]{1,20}$') { throw 'Invalid evidence suffix' }
$campaignRoot = $PSScriptRoot
$pythonRuntime = Join-Path $campaignRoot '..\qwen-strata-20260926\.venv\Scripts\python.exe'
Set-Location -LiteralPath $campaignRoot
foreach ($tierContext in @(16384, 32768, 65536)) {
    $tierConfig = "configs/tier$tierContext.json"
    foreach ($trialMode in @('load', 'smoke')) {
        $trialTag = "$trialMode$tierContext$Suffix"
        if (Test-Path -LiteralPath (Join-Path $campaignRoot "runs/$trialTag")) {
            throw "Evidence tag already exists: $trialTag"
        }
        & $pythonRuntime '.\agent_guard.py' $trialMode --config $tierConfig --tag $trialTag --timeout 900
        if ($LASTEXITCODE -ne 0) { throw "$trialTag failed; no higher tier will start" }
    }
}
