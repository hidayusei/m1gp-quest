@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0StartDemo.ps1"
echo.
echo Demo launcher closed.
pause
