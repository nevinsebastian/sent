#!/bin/bash

echo "🚀 Starting AI + React Feedback System..."

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $1 is already in use"
        return 1
    else
        return 0
    fi
}

# Check if backend port is available
if check_port 5001; then
    echo "✅ Port 5001 is available for backend"
else
    echo "❌ Please stop the process using port 5001"
    exit 1
fi

# Check if frontend port is available
if check_port 3000; then
    echo "✅ Port 3000 is available for frontend"
else
    echo "❌ Please stop the process using port 3000"
    exit 1
fi

echo ""
echo "📋 Starting services..."
echo ""

# Start backend in background
echo "🔧 Starting Flask backend..."
cd backend
python app.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend in background
echo "⚛️  Starting React frontend..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "🎉 Both services are starting..."
echo ""
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend:  http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop both services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait 