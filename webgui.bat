@echo off

set PYTHON_VER=3.9.18
set PARENT_DIR=%CD%

:: Check if Python version meets the recommended version
python --version 2>nul | findstr /b /c:"Python %PYTHON_VER%" >nul
if errorlevel 1 (
    echo Warning: Python version %PYTHON_VER% is recommended.
)


IF NOT EXIST venv (
    echo Creating venv...
    python -m venv venv
)


:: Create the directory if it doesn't exist
mkdir ".\logs\setup" > nul 2>&1

:: install requirements
python.exe -m pip install -r requirements.txt

:: change directory to main
chdir ".\SystemCode\src\main"
start cmd /k "flask run"

chdir /d %PARENT_DIR%


:: Deactivate the virtual environment
call .\venv\Scripts\deactivate.bat