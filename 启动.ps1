[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$projectLauncher = Join-Path $PSScriptRoot 'scripts\start_project.py'

if (-not (Test-Path -LiteralPath $projectPython -PathType Leaf)) {
    throw 'Project Python environment is missing. See README.md.'
}

& $projectPython -X utf8 $projectLauncher
exit $LASTEXITCODE
