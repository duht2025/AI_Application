# Create_MulAg_env.ps1
param(
    [string]$EnvName = "multi_agent_clean"
)

Write-Host "Creating new virtual environment: $EnvName" -ForegroundColor Green
python -m venv $EnvName

$activateScript = "$EnvName\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "Activated $EnvName" -ForegroundColor Yellow
} else {
    Write-Host "Activation failed. Please activate manually." -ForegroundColor Red
    exit 1
}

Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

if (Test-Path "MulAgrequires.txt") {
    Write-Host "Installing packages from MulAgrequires.txt..." -ForegroundColor Yellow
    pip install -r MulAgrequires.txt
    Write-Host "Installation completed successfully!" -ForegroundColor Green
} else {
    Write-Host "MulAgrequires.txt not found!" -ForegroundColor Red
    exit 1
}

Write-Host "To activate the new environment later, run: .\$EnvName\Scripts\Activate.ps1"