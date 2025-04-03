@echo off
setlocal enabledelayedexpansion

:: Check if the script has already restarted
if "%RESTARTED%"=="1" goto SKIP_RESTART

:: Flag to track restart requirement
set RESTART_REQUIRED=0

:: Display header
echo.
echo ==================================================
echo               PROJECT SETUP INITIALIZED
echo ==================================================
echo.

:: STEP 1: CHECK AND INSTALL PYENV
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (-not (Get-Command pyenv -ErrorAction SilentlyContinue)) { Write-Host 'Installing pyenv-win...'; Invoke-WebRequest -UseBasicParsing -Uri 'https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1' -OutFile './install-pyenv-win.ps1'; & './install-pyenv-win.ps1'; exit 10; } else { Write-Host 'pyenv is already installed.'; exit 0 }"

:: Check the exit code from PowerShell (10 means pyenv was installed)
if %ERRORLEVEL%==10 set RESTART_REQUIRED=1

:: STEP 1B: RESTART IF REQUIRED
if "%RESTART_REQUIRED%"=="1" (
    echo Restarting command prompt to apply changes...
    set RESTARTED=1
    call "%~f0"
    exit /b
)

:SKIP_RESTART

:: STEP 2: INSTALL PYTHON 3.10.11 USING PYENV
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (-not (pyenv versions | Select-String '3.10.11')) { Write-Host 'Installing Python 3.10.11...'; pyenv install 3.10.11; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; Write-Host 'Python 3.10.11 installed successfully.'; } else { Write-Host 'Python 3.10.11 is already installed.'; }; Write-Host 'Setting local Python version to 3.10.11...'; pyenv local 3.10.11; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; Write-Host 'Local Python version set.';"
echo Python installation completed.
echo.

:: STEP 3: CREATE VIRTUAL ENVIRONMENT
powershell -NoProfile -ExecutionPolicy Bypass -Command "$venvPath=(Get-Location).Path+'\\BaseEnv'; if (-not (Test-Path $venvPath)) { Write-Host 'Creating virtual environment...'; pyenv exec python -m venv $venvPath; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; Write-Host 'Virtual environment created successfully.'; } else { Write-Host 'Virtual environment already exists.'; }"
echo Virtual environment setup completed.
echo.

:: STEP 4: INSTALL REQUIRED PACKAGES
powershell -NoProfile -ExecutionPolicy Bypass -Command "$venvPython=Join-Path (Get-Location) 'BaseEnv\\Scripts\\python.exe'; $reqFile=Join-Path (Get-Location) 'requirements.txt'; Write-Host 'Upgrading pip...'; & $venvPython -m pip install --upgrade pip; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; if (Test-Path $reqFile) { Write-Host 'Installing required dependencies...'; & $venvPython -m pip install -r $reqFile; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; Write-Host 'All dependencies installed successfully.'; } else { Write-Host 'requirements.txt not found.'; exit 1 }"
echo Dependencies installed successfully.
echo.

:: STEP 5: LAUNCH AI MODULE
echo ==================================================
echo               LAUNCHING AI MODULE
echo ==================================================
BaseEnv\Scripts\python.exe -m InsightEX
echo.
pause
