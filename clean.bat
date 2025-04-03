@echo off
setlocal enabledelayedexpansion

:: Check if the script has already restarted
if "%RESTARTED%"=="1" goto SKIP_RESTART

:: Flag to track restart requirement
set RESTART_REQUIRED=0

:: Display header
echo.
echo ==================================================
echo           PROJECT CLEANUP INITIALIZED
echo ==================================================
echo.

:: STEP 1: DELETE VIRTUAL ENVIRONMENT
if exist "BaseEnv" (
    echo Deleting virtual environment...
    rmdir /s /q "BaseEnv"
    if %ERRORLEVEL% neq 0 (
        echo Error deleting virtual environment.
        exit /b 1
    ) else (
        echo Virtual environment deleted successfully.
    )
) else (
    echo Virtual environment not found.
)
echo.

:: STEP 2: UNINSTALL PYTHON 3.10.11 USING PYENV
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (-not (Get-Command pyenv -ErrorAction SilentlyContinue)) { Write-Host 'pyenv is not installed. Skipping Python uninstallation.'; exit 0 } elseif (-not (pyenv versions | Select-String '3.10.11')) { Write-Host 'Python 3.10.11 is not installed. Skipping uninstallation.'; exit 0 } else { Write-Host 'Uninstalling Python 3.10.11...'; pyenv uninstall -f 3.10.11; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } else { Write-Host 'Python 3.10.11 uninstalled successfully.' } }"
if %ERRORLEVEL% neq 0 (
    echo Error uninstalling Python 3.10.11.
    exit /b 1
)
echo.

:: STEP 3: UNINSTALL PYENV-WIN
:: Typically, pyenv-win is installed in %USERPROFILE%\.pyenv. Adjust if necessary.
set "PYENV_PATH=%USERPROFILE%\.pyenv"
if exist "%PYENV_PATH%" (
    echo Deleting pyenv-win installation...
    rmdir /s /q "%PYENV_PATH%"
    if %ERRORLEVEL% neq 0 (
        echo Error deleting pyenv-win installation.
        exit /b 1
    ) else (
        echo pyenv-win uninstalled successfully.
    )
) else (
    echo pyenv-win installation not found.
)
echo.

:: STEP 4: REMOVE PYENV-WIN PATHS FROM THE USER'S ENVIRONMENT VARIABLE
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = [Environment]::GetEnvironmentVariable('PATH','User'); $remove = [Environment]::ExpandEnvironmentVariables('%USERPROFILE%\.pyenv\pyenv-win\bin'); if($p -like '*'+$remove+'*') { $new = $p -replace [regex]::Escape($remove + ';'), ''; [Environment]::SetEnvironmentVariable('PATH', $new, 'User'); Write-Host 'Removed pyenv-win bin from PATH.' } else { Write-Host 'pyenv-win bin not found in PATH.' }"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = [Environment]::GetEnvironmentVariable('PATH','User'); $remove = [Environment]::ExpandEnvironmentVariables('%USERPROFILE%\.pyenv\pyenv-win\shims'); if($p -like '*'+$remove+'*') { $new = $p -replace [regex]::Escape($remove + ';'), ''; [Environment]::SetEnvironmentVariable('PATH', $new, 'User'); Write-Host 'Removed pyenv-win shims from PATH.' } else { Write-Host 'pyenv-win shims not found in PATH.' }"
echo.

:: STEP 5: RESTART COMMAND PROMPT IF NEEDED
if "%RESTART_REQUIRED%"=="1" (
    echo Restarting command prompt to apply changes...
    set RESTARTED=1
    call "%~f0"
    exit /b
)

:SKIP_RESTART

echo ==================================================
echo             PROJECT CLEANUP COMPLETED
echo ==================================================
pause
