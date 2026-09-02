param([switch]$Clean)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Spec = Join-Path $ProjectRoot "TechnicalDocumentIntelligence.spec"
$Distribution = Join-Path $ProjectRoot "dist\TechnicalDocumentIntelligence"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project virtual-environment Python was not found: $Python"
}

$arguments = @("-m", "PyInstaller", "--noconfirm")
if ($Clean) { $arguments += "--clean" }
$arguments += $Spec
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

New-Item -ItemType Directory -Force -Path (Join-Path $Distribution "config") | Out-Null
Copy-Item -LiteralPath (Join-Path $ProjectRoot "config\config.toml") -Destination (Join-Path $Distribution "config\config.toml") -Force
New-Item -ItemType Directory -Force -Path (Join-Path $Distribution "models\embedding\bge-m3") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Distribution "models\llm\qwen3-8b") | Out-Null

Write-Host "Built: $Distribution"
Write-Host "Copy external model files into the prepared models directories before use."
