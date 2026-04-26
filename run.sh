#!/bin/bash

# URL Detection API Startup Script
# Automatically starts the FastAPI server and opens the web UI

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add home bin to PATH if not already there
export PATH="$HOME/bin:$PATH"

# Change to the project directory
cd "$SCRIPT_DIR"

echo "🚀 Starting URL Detection API..."
echo "📁 Project directory: $SCRIPT_DIR"

# Check if micromamba is available
if ! command -v micromamba &> /dev/null; then
    echo "❌ Error: micromamba is not installed or not in PATH"
    echo "Please install micromamba or ensure it's available in your PATH"
    exit 1
fi

# Check if the url_detection environment exists
if ! micromamba env list | grep -q url_detection; then
    echo "❌ Error: 'url_detection' micromamba environment not found"
    echo "Please create the environment with: micromamba create -n url_detection -c conda-forge python=3.11"
    exit 1
fi

# Start the FastAPI server
echo "🔧 Starting FastAPI server (Ctrl+C to stop)..."
echo "📡 Server will be available at: http://127.0.0.1:8000"
echo "🌐 Web UI will open at: http://127.0.0.1:8000/app"
echo ""

# Run the server in the background to open the browser
micromamba run -n url_detection python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &

# Store the server PID
SERVER_PID=$!

# Wait a moment for the server to start
sleep 3

# Check if the server started successfully
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Failed to start the server"
    exit 1
fi

# Try to open the browser (works on macOS, Linux, and Windows)
if command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open http://127.0.0.1:8000/app &
elif command -v open &> /dev/null; then
    # macOS
    open http://127.0.0.1:8000/app &
else
    echo "✅ Server started successfully!"
    echo "📝 Note: Please manually open http://127.0.0.1:8000/app in your browser"
fi

echo "✅ API is running!"
echo ""
echo "API Health: http://127.0.0.1:8000/health"
echo "API Docs:   http://127.0.0.1:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

# Keep the script running and forward signals to the server
wait $SERVER_PID
