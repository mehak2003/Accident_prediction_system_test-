@echo off
echo ========================================
echo  AcciHotspot Dashboard - Starting...
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo [1/4] Installing Backend Dependencies...
cd backend
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
pip install -r requirements.txt >nul 2>&1
echo     Backend dependencies installed!

echo.
echo [2/4] Installing Frontend Dependencies...
cd ..\frontend
if not exist "node_modules" (
    call npm install
) else (
    echo     Frontend dependencies already installed!
)

echo.
echo [3/4] Starting Backend Server...
cd ..\backend
start "AcciHotspot Backend" cmd /k "venv\Scripts\activate && python app.py"
timeout /t 3 /nobreak >nul

echo.
echo [4/4] Starting Frontend Server...
cd ..\frontend
start "AcciHotspot Frontend" cmd /k "npm run dev"

echo.
echo ========================================
echo  Servers Starting!
echo ========================================
echo.
echo  Backend:  http://localhost:5000
echo  Frontend: http://localhost:5173
echo.
echo  Two new windows will open:
echo  1. Backend (Flask API)
echo  2. Frontend (React App)
echo.
echo  Wait 5 seconds, then open:
echo  http://localhost:5173
echo.
echo  To upload data:
echo  1. Go to "Data" page
echo  2. Upload CSV file
echo  3. Click "Run Full Pipeline"
echo  4. Wait 2-5 minutes
echo  5. View results on Dashboard
echo.
echo ========================================
timeout /t 5 /nobreak >nul
start http://localhost:5173
pause
