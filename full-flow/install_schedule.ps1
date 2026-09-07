param(
    [switch]$Install,
    [string]$At = '08:30',
    [string]$Python = '',
    [ValidateSet('openai', 'claude', 'google', 'mixed')]
    [string]$Provider = 'openai',
    [string]$TaskName = 'PersonalStockResearchDaily'
)
$ErrorActionPreference = 'Stop'
if (-not $Python) { $Python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe' }
$Python = (Resolve-Path -LiteralPath $Python).Path
$runner = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'run_daily.ps1')).Path
$time = [datetime]::ParseExact($At, 'HH:mm', [Globalization.CultureInfo]::InvariantCulture)
Write-Output "Daily report at $At in Windows timezone: $((Get-TimeZone).Id)"
Write-Output "Python: $Python; provider: $Provider; reports: $PSScriptRoot\reports"
if (-not $Install) {
    Write-Output 'Preview only. Run with -Install to register this scheduled task.'
    exit 0
}
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw 'A task with this name exists. Choose another TaskName or manage it in Task Scheduler.'
}
if ($Python.Contains('"') -or $runner.Contains('"')) { throw 'Invalid path quoting' }
$arguments = '-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $runner + '" -Python "' + $Python + '" -Provider ' + $Provider
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -Daily -At $time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 3)
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Description 'Generate local daily stock research; no trades or email.' | Out-Null
Write-Output 'Task registered. Check account logon conditions and power settings in Windows Task Scheduler.'
