@echo off
rem Paralic launcher for Windows: creates a virtual environment on first run,
rem installs the dependencies, then starts the server and opens the browser.
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  set "PY=python"
)

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
if errorlevel 1 (
  echo Python 3.9 or newer is required: https://www.python.org/downloads/
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment ^(.venv^)...
  %PY% -m venv .venv || goto :error
)

if not exist ".venv\.installed" (
  echo Installing dependencies ^(first run only, this can take a minute^)...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
  type nul > ".venv\.installed"
)

".venv\Scripts\python.exe" -m paralic %*
exit /b %errorlevel%

:error
echo.
echo Setup failed - see the messages above.
pause
exit /b 1
