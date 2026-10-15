@echo off
rem Windows double-click entry for qvix.command (same thing, run inside WSL).
rem Deps live in the WSL venv; the Windows python.exe is just the Store stub.
rem qvix.command finds the venv and holds the window open for a keypress.
rem bash -l: login shell, so ~/.local/bin (gh, for the threshold sync) is on PATH.
rem Keep this file ASCII + CRLF: cmd.exe parses .bat in the console codepage.
chcp 65001 >nul
wsl.exe --cd "%~dp0." bash -l ./qvix.command
if errorlevel 1 (
    echo.
    echo qvix failed, exit code %errorlevel%
    pause
)
