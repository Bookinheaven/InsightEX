# Requires PowerShell to be run as Administrator

# Step 1: Check if pyenv is installed, if not, install it
if (-not (Get-Command pyenv -ErrorAction SilentlyContinue)) {
    Write-Host "🔹 Installing pyenv-win..."

    # Download and execute the official pyenv-win installation script
    Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"
    &"./install-pyenv-win.ps1"

    # Add pyenv to the system path
    $pyenvRoot = "$env:USERPROFILE\.pyenv"
    [System.Environment]::SetEnvironmentVariable("PYENV", $pyenvRoot, [System.EnvironmentVariableTarget]::User)
    [System.Environment]::SetEnvironmentVariable("PYENV_ROOT", $pyenvRoot, [System.EnvironmentVariableTarget]::User)
    [System.Environment]::SetEnvironmentVariable("PATH", "$pyenvRoot\bin;$pyenvRoot\shims;$env:PATH", [System.EnvironmentVariableTarget]::User)

    # Reload the environment variables
    $env:PYENV = $pyenvRoot
    $env:PYENV_ROOT = $pyenvRoot
    $env:PATH = "$pyenvRoot\bin;$pyenvRoot\shims;$env:PATH"

    Write-Host "✅ pyenv installed successfully! Restart your terminal for changes to take effect."
} else {
    Write-Host "✅ pyenv is already installed."
}

# Step 2: Install Python 3.10.11 if not installed
$pythonVersion = "3.10.11"
if (-not (pyenv versions | Select-String $pythonVersion)) {
    Write-Host "🔹 Installing Python $pythonVersion using pyenv..."
    pyenv install $pythonVersion
} else {
    Write-Host "✅ Python $pythonVersion is already installed."
}

# Set the Python version globally
pyenv global $pythonVersion

# Step 3: Define environment paths
$BaseEnvDir = "$env:USERPROFILE\.BaseEnv"
$VenvDir = "$BaseEnvDir\.OpenVino"

# Step 4: Create base environment directory if it doesn't exist
if (-not (Test-Path $BaseEnvDir)) {
    Write-Host "🔹 Creating base environment directory at $BaseEnvDir..."
    New-Item -ItemType Directory -Path $BaseEnvDir | Out-Null
}

# Step 5: Create virtual environment if it doesn't exist
if (-not (Test-Path $VenvDir)) {
    Write-Host "🔹 Creating OpenVINO virtual environment in $VenvDir..."
    python -m venv $VenvDir
} else {
    Write-Host "✅ Virtual environment already exists. Skipping creation."
}

# Step 6: Activate the virtual environment
Write-Host "🔹 Activating OpenVINO environment..."
$ActivateScript = "$VenvDir\Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    & $ActivateScript
} else {
    Write-Host "❌ Failed to activate the virtual environment. Check the installation."
    exit 1
}

# Step 7: Upgrade pip and install dependencies
Write-Host "🔹 Upgrading pip..."
pip install --upgrade pip

Write-Host "🔹 Installing OpenVINO and other dependencies..."
pip install openvino openvino-dev ultralytics torch opencv-python numpy

# Step 8: Verify installation
Write-Host "🔹 Verifying installation..."
python -c "import openvino; import torch; import cv2; print('✅ OpenVINO Environment Setup Complete!')"

Write-Host "🎉 OpenVINO Environment is set up successfully!"
