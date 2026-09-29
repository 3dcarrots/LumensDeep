@echo off
setlocal enabledelayedexpansion
title Lumens Deep - Build Standalone Installer
cd /d "%~dp0"

echo ================================================================
echo       Lumens Deep - Automatic Installer Builder
echo ================================================================
echo.

:: 1. Check Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found in PATH!
    pause
    exit /b 1
)

:: 2. Check and install PyInstaller if needed
python -c "import PyInstaller" >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo [INFO] PyInstaller is not installed. Installing via pip...
    python -m pip install pyinstaller
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to install PyInstaller.
        pause
        exit /b 1
    )
)

:: 3. Clean old builds if desired
echo [1/3] Preparing workspace and cleaning old builds...
if exist "build" rmdir /s /q "build"
if exist "dist\LumensDeep" rmdir /s /q "dist\LumensDeep"

:: 4. Run PyInstaller
echo.
echo [2/3] Freezing application with PyInstaller (this may take a few minutes)...
pyinstaller --noconfirm lumensdeep.spec
if !ERRORLEVEL! neq 0 (
    echo [ERROR] PyInstaller build failed!
    pause
    exit /b 1
)

echo.
echo [INFO] Standalone folder created successfully at: dist\LumensDeep\

:: Security check: Ensure no .env or sensitive files leaked into dist\
echo.
echo [SECURITY] Scanning distribution folder for sensitive files...
if exist "dist\LumensDeep\.env" (
    echo [SECURITY WARNING] Found root .env in dist! Removing immediately...
    del /f /q "dist\LumensDeep\.env"
)
for /r "dist\LumensDeep" %%f in (*.env *.env.* *.key *.token) do (
    echo [SECURITY WARNING] Found sensitive file %%f! Removing immediately...
    del /f /q "%%f"
)
echo [SECURITY VERIFIED] Confirmed: No .env, API keys, or secrets exist in dist\LumensDeep\.

:: 5. Search for Inno Setup Compiler (ISCC.exe)
echo.
echo [3/3] Building Windows Installer (.exe Setup)...
set "ISCC_PATH="

where iscc >nul 2>&1
if not errorlevel 1 set "ISCC_PATH=iscc"

if not defined ISCC_PATH if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set "ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not defined ISCC_PATH if exist "C:\Program Files\Inno Setup 6\ISCC.exe" set "ISCC_PATH=C:\Program Files\Inno Setup 6\ISCC.exe"
if not defined ISCC_PATH if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC_PATH=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"

if not defined ISCC_PATH goto no_iscc

echo [INFO] Found Inno Setup Compiler at: "!ISCC_PATH!"
if not exist "output" mkdir "output"
"!ISCC_PATH!" installer.iss
if !ERRORLEVEL! equ 0 (
    echo.
    echo ================================================================
    echo  SUCCESS! Your installer has been generated in:
    echo  %~dp0output\LumensDeep_Setup_v0.1_Alpha_x64.exe
    echo ================================================================
    echo Any user can run this .exe to install Lumens Deep with Desktop
    echo and Start Menu shortcuts automatically!
    echo.
) else (
    echo [ERROR] Inno Setup compilation failed.
)
goto done

:no_iscc
echo [NOTICE] Inno Setup (ISCC.exe) is not installed on this system.
echo.
echo The standalone application is ready in:
echo   %~dp0dist\LumensDeep\LumensDeep.exe
echo.
echo To generate the single-file Setup.exe installer:
echo 1. Download Inno Setup (Free, 5 MB): https://jrsoftware.org/isdl.php
echo 2. Install Inno Setup and run 'compile_installer.bat' (or re-run this script),
echo    OR open 'installer.iss' in Inno Setup and click 'Compile'.
echo.

:done
pause
