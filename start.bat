@echo off
REM MOM-Bot System Startup Script (Windows)
REM Start both API and Web UI simultaneously

echo.
echo 🚀 Starting MOM-Bot System...
echo.

REM Check if we're in the right directory
if not exist "requirements.txt" (
    echo ❌ Error: Please run this script from the MOM-Bot root directory
    exit /b 1
)

echo 📦 Checking Python environment...
python --version

echo.
echo 🌐 Starting REST API Server...
echo    Running on: http://localhost:5000
start cmd /k python src/api/server.py

timeout /t 2

echo.
echo 📊 Starting Streamlit Web App...
echo    Running on: http://localhost:8501
start cmd /k streamlit run app.py

timeout /t 3

echo.
echo ✅ System Started Successfully!
echo.
echo Available Services:
echo   🌐 REST API:     http://localhost:5000
echo   📊 Web UI:       http://localhost:8501
echo   📄 API Docs:     http://localhost:5000/
echo.
echo Close these windows to stop the services.
