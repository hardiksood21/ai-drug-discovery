@echo off
echo =========================================================
echo    AI + Drug Discovery Portfolio - Push to GitHub
echo =========================================================
echo Repository: https://github.com/hardiksood21/ai-drug-discovery.git
echo.

git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not in PATH.
    echo Please install Git for Windows from: https://git-scm.com/download/win
    echo Or install GitHub Desktop from: https://desktop.github.com/
    pause
    exit /b 1
)

echo Initializing git repository...
git init
git add .
git commit -m "Initial portfolio commit with Phase 1 Delaney ESOL baseline project"
git branch -M main
git remote add origin https://github.com/hardiksood21/ai-drug-discovery.git

echo.
echo Pushing to GitHub (https://github.com/hardiksood21/ai-drug-discovery.git)...
git push -u origin main

echo.
echo =========================================================
echo Done! Your portfolio is live on GitHub.
echo =========================================================
pause
