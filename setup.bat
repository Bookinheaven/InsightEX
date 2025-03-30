@echo off
echo.
echo =================== Starting Setup ============================
echo.

REM ----------------------------------------------------------------
REM Step 1: Install pyenv-win if not already installed
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (-not (Get-Command pyenv -ErrorAction SilentlyContinue)) { Write-Host 'Installing pyenv-win...' -ForegroundColor Cyan; Invoke-WebRequest -UseBasicParsing -Uri 'https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1' -OutFile './install-pyenv-win.ps1'; & './install-pyenv-win.ps1'; $pyenvRoot = $env:USERPROFILE + '\\.pyenv'; [System.Environment]::SetEnvironmentVariable('PYENV', $pyenvRoot, [System.EnvironmentVariableTarget]::User); [System.Environment]::SetEnvironmentVariable('PYENV_ROOT', $pyenvRoot, [System.EnvironmentVariableTarget]::User); $newPath = $pyenvRoot + '\\bin;' + $pyenvRoot + '\\shims;' + $env:PATH; [System.Environment]::SetEnvironmentVariable('PATH', $newPath, [System.EnvironmentVariableTarget]::User); $env:PYENV = $pyenvRoot; $env:PYENV_ROOT = $pyenvRoot; $env:PATH = $newPath; Write-Host 'pyenv installed successfully!' -ForegroundColor Green; } else { Write-Host 'pyenv is already installed.' -ForegroundColor Yellow; }"
echo ------------------ pyenv-win step complete --------------------
echo.

REM ----------------------------------------------------------------
REM Step 1b: Refresh environment variables so pyenv is immediately available
REM Try to call refreshenv (from Chocolatey). If that fails, fallback to a PowerShell command.
call refreshenv 2>NUL || powershell -NoProfile -ExecutionPolicy Bypass -Command "[System.Environment]::SetEnvironmentVariable('PATH', [System.Environment]::GetEnvironmentVariable('Path','User') + ';' + [System.Environment]::GetEnvironmentVariable('Path','Machine'), [System.EnvironmentVariableTarget]::Process)"
echo Environment variables refreshed.
echo.

REM ----------------------------------------------------------------
REM Step 2: Install Python 3.10.11 using pyenv and set it as the local version
powershell -NoProfile -ExecutionPolicy Bypass -Command "$pythonVersion = '3.10.11'; if (-not (pyenv versions | Select-String $pythonVersion)) { Write-Host ('Installing Python ' + $pythonVersion + ' using pyenv...') -ForegroundColor Cyan; pyenv install $pythonVersion; Write-Host ('Python ' + $pythonVersion + ' installed successfully.') -ForegroundColor Green; } else { Write-Host ('Python ' + $pythonVersion + ' is already installed.') -ForegroundColor Yellow; } ; Write-Host ('Setting local Python version to ' + $pythonVersion + '...') -ForegroundColor Cyan; pyenv local $pythonVersion; Write-Host 'Local Python version set.' -ForegroundColor Green;"
echo ---------------- Python Setup Step Complete ----------------------
echo.

REM ----------------------------------------------------------------
REM Step 3: Create a virtual environment with Python 3.10
powershell -NoProfile -ExecutionPolicy Bypass -Command "$venvPath = (Get-Location).Path + '\\BaseEnv'; if (-not (Test-Path $venvPath)) { Write-Host 'Creating virtual environment in ' + $venvPath -ForegroundColor Cyan; pyenv exec python3.10 -m venv $venvPath; Write-Host 'Virtual environment created successfully.' -ForegroundColor Green; } else { Write-Host 'Virtual environment BaseEnv already exists at ' + $venvPath -ForegroundColor Yellow; }"
echo ---------------- Virtual Environment Creation Complete -----------
echo.

REM ----------------------------------------------------------------
REM Step 4: Activate the virtual environment and install required modules using Python 3.10
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $venvPath = Join-Path (Get-Location) 'BaseEnv'; Write-Host 'Activating virtual environment at ' $venvPath -ForegroundColor Cyan; & (Join-Path $venvPath 'Scripts\Activate.ps1'); Write-Host 'Upgrading pip...' -ForegroundColor Cyan; & (Join-Path $venvPath 'Scripts\python.exe') -m pip install --upgrade pip; $reqFile = Join-Path (Get-Location) 'requirements.txt'; if (Test-Path $reqFile) { Write-Host 'Installing required modules from requirements.txt...' -ForegroundColor Cyan; & (Join-Path $venvPath 'Scripts\python.exe') -m pip install -r $reqFile; Write-Host 'All required modules have been installed.' -ForegroundColor Green; } else { Write-Host 'requirements.txt not found in the current directory.' -ForegroundColor Red; } }"
echo =================== Setup Complete! ============================
echo.

REM ----------------------------------------------------------------
REM Step 5: Launch the AI Module
echo =================== Launching AI Module ======================
echo.
BaseEnv\Scripts\python.exe -m InsightEX
echo.

pause
