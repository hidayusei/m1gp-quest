@echo off
setlocal
cd /d "%~dp0"
title M1 QUEST Demo Launcher
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0StartDevDemo.ps1"
if errorlevel 1 (
  echo.
  echo Launcher stopped with an error.
  pause
)
endlocal
