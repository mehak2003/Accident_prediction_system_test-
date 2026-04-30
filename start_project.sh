#!/bin/bash

echo "========================================"
echo " AcciHotspot Dashboard - Starting..."
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed"
    echo "Please install Python3 from https://www.python.org/"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js is not installed"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

echo "[1/4] Installing Backend Dependencies..."
cd backend
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt > /dev/null 2>&1
echo "    Backend dependencies installed!"

echo ""
echo "[2/4] Installing Frontend Dependencies..."
cd ../frontend
if [ ! -d "node_modules" ]; then
    npm install
else
    echo "    Frontend dependencies already installed!"
fi

echo ""
echo "[3/4] Starting Backend Server..."
cd ../backend
source venv/bin/activate
# Start backend in background
python3 app.py > backend.log 2>&1 &
BACKEND_PID=$!
echo "    Backend started (PID: $BACKEND_PID)"
sleep 3

echo ""
echo "[4/4] Starting Frontend Server..."
cd ../frontend
# Start frontend in background
npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "    Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "========================================"
echo " Servers Running!"
echo "========================================"
echo ""
echo "  Backend:  http://localhost:5000"
echo "  Frontend: http://localhost:5173"
echo ""
echo "  Process IDs:"
echo "  - Backend:  $BACKEND_PID"
echo "  - Frontend: $FRONTEND_PID"
echo ""
echo "  Logs:"
echo "  - Backend:  backend/backend.log"
echo "  - Frontend: frontend/frontend.log"
echo ""
echo "  To upload data:"
echo "  1. Open http://localhost:5173"
echo "  2. Go to 'Data' page"
echo "  3. Upload CSV file"
echo "  4. Click 'Run Full Pipeline'"
echo "  5. Wait 2-5 minutes"
echo "  6. View results on Dashboard"
echo ""
echo "  To stop servers:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "========================================"

# Wait a bit then open browser
sleep 5
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:5173
elif command -v open &> /dev/null; then
    open http://localhost:5173
fi

# Keep script running
echo ""
echo "Press Ctrl+C to stop all servers..."
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
