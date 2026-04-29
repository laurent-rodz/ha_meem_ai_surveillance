@echo off
echo ==================================================
echo   Git and GitHub Logout Script (Public Desktop)
echo ==================================================
echo.

echo [1/3] Removing Git credentials from Windows Credential Manager...
cmdkey /delete:LegacyGeneric:target=git:https://github.com >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo - Successfully removed github.com credentials.
) else (
    echo - No github.com credentials found or already removed.
)

echo.
echo [2/3] Unsetting global Git user configuration...
git config --global --unset user.name >nul 2>&1
git config --global --unset user.email >nul 2>&1
echo - Global Git user.name and user.email unset.

echo.
echo [3/3] Checking for GitHub CLI (gh)...
where gh >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo - GitHub CLI found. Logging out...
    echo y | gh auth logout --hostname github.com >nul 2>&1
    echo - Logged out of GitHub CLI.
) else (
    echo - GitHub CLI not installed. Skipping.
)

echo.
echo ==================================================
echo   Logout Complete. You can now safely close this.
echo ==================================================
pause