@echo off
title Lumens Deep - Compile Inno Setup Installer
cd /d "%~dp0"

echo ================================================================
echo       Lumens Deep - Compile Inno Setup (.exe Installer)
echo ================================================================
echo.

if not exist "dist\LumensDeep\LumensDeep.exe" (
    echo [ERROR] dist\LumensDeep\LumensDeep.exe not found!
    echo Please run build_installer.bat first to freeze the application.
    pause
    exit /b 1
)

:: Search for ISCC.exe
set "ISCC_PATH="
where iscc >nul 2>&1
if not errorlevel 1 set "ISCC_PATH=iscc"

if not defined ISCC_PATH if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set "ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not defined ISCC_PATH if exist "C:\Program Files\Inno Setup 6\ISCC.exe" set "ISCC_PATH=C:\Program Files\Inno Setup 6\ISCC.exe"
if not defined ISCC_PATH if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC_PATH=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"

if not defined ISCC_PATH goto no_iscc

echo [INFO] Found Inno Setup Compiler at: "%ISCC_PATH%"
if not exist "output" mkdir "output"
"%ISCC_PATH%" installer.iss
if not errorlevel 1 (
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
echo 2. Install Inno Setup and run this script (compile_installer.bat),
echo    OR open 'installer.iss' in Inno Setup and click 'Compile'.
echo.

:done
pause
