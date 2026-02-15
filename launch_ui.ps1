$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot

$pythonExe = if ($env:TINYLLM_PYTHON) {
    $env:TINYLLM_PYTHON
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    (Get-Command python).Source
} else {
    throw 'Python was not found. Activate the tinyLLM environment or set TINYLLM_PYTHON.'
}

$modelConfig = Join-Path $projectDir 'configs\models.json'
if (-not (Test-Path -LiteralPath $modelConfig)) {
    Copy-Item -LiteralPath (Join-Path $projectDir 'configs\models.example.json') -Destination $modelConfig
    throw 'Created configs\models.json. Add the downloaded model files, then run this script again.'
}

$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = $projectDir
Push-Location -LiteralPath $projectDir
try {
    & $pythonExe -u chat_server.py --port 8501
} finally {
    Pop-Location
}
