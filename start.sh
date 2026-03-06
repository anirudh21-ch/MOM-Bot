#!/bin/bash
# MOM-Bot System Startup Script
# Start both API and Web UI simultaneously

echo "🚀 Starting MOM-Bot System..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the MOM-Bot root directory"
    exit 1
fi

echo "${BLUE}📦 Checking Python environment...${NC}"
python --version

echo ""
echo "${BLUE}🌐 Starting REST API Server...${NC}"
echo "   Running on: http://localhost:5000"
python src/api/server.py &
API_PID=$!

sleep 2

echo ""
echo "${BLUE}📊 Starting Streamlit Web App...${NC}"
echo "   Running on: http://localhost:8501"
streamlit run app.py &
STREAMLIT_PID=$!

sleep 3

echo ""
echo "${GREEN}✅ System Started Successfully!${NC}"
echo ""
echo "Available Services:"
echo "  🌐 REST API:     http://localhost:5000"
echo "  📊 Web UI:       http://localhost:8501"
echo "  📄 API Docs:     http://localhost:5000/"
echo ""
echo "Press Ctrl+C to stop all services..."
echo ""

# Wait for both processes
wait $API_PID $STREAMLIT_PID
