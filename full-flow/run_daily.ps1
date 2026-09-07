param(
    [string]$Python = '',
    [ValidateSet('openai', 'claude', 'google', 'mixed')]
    [string]$Provider = 'openai'
)
$ErrorActionPreference = 'Stop'
if (-not $Python) { $Python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe' }
$Python = (Resolve-Path -LiteralPath $Python).Path
$reportDirectory = Join-Path $PSScriptRoot 'reports'
New-Item -ItemType Directory -Force -Path $reportDirectory | Out-Null
$logFile = Join-Path $reportDirectory ('run-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
& $Python (Join-Path $PSScriptRoot 'main.py') --provider $Provider 2>&1 |
    Tee-Object -FilePath $logFile
exit $LASTEXITCODE
