@echo off
REM start.bat - SafeRoute Quick Start Script for Windows

echo ========================================
echo 🚀 SafeRoute - Starting Application...
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH!
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo ✅ Python found: 
python --version

REM Check if virtual environment exists
if not exist "venv\" (
    echo.
    echo ❌ Virtual environment not found!
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created!
)

REM Activate virtual environment
echo.
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if Flask is installed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo.
    echo ❌ Dependencies not installed!
    echo Installing dependencies (this may take a few minutes)...
    pip install --upgrade pip
    pip install -r requirements_web.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies!
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed!
) else (
    echo ✅ Dependencies already installed!
)

REM Check if model exists
echo.
if not exist "random_forest_model.pkl" (
    echo ❌ Model file not found!
    echo Training model... (this may take a few minutes)
    python train_random_forest.py
    if errorlevel 1 (
        echo ❌ Failed to train model!
        pause
        exit /b 1
    )
    echo ✅ Model trained successfully!
) else (
    echo ✅ Model file found!
)

REM Check if data file exists
if not exist "cleaned_chicago_crime_data.csv" (
    echo.
    echo ❌ Crime data file not found!
    echo Please ensure 'cleaned_chicago_crime_data.csv' is in the project directory.
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Crime data file found!
)

REM Create directories if they don't exist
echo.
echo 📁 Creating required directories...
if not exist "templates\" mkdir templates
if not exist "static\" mkdir static
if not exist "static\css\" mkdir static\css
if not exist "static\js\" mkdir static\js
echo ✅ Directories ready!

REM Check if templates exist
echo.
if not exist "templates\index.html" (
    echo ⚠️  Warning: templates\index.html not found!
    echo Please make sure index.html is in the templates folder.
)

if not exist "templates\map.html" (
    echo ⚠️  Warning: templates\map.html not found!
    echo Please make sure map.html is in the templates folder.
)

REM Start the application
echo.
echo ========================================
echo ✨ Starting SafeRoute Web Server...
echo ========================================
echo.
echo 🌐 The application will be available at:
echo    http://localhost:5000
echo.
echo 📝 Instructions:
echo    1. Wait for the server to start
echo    2. Open your browser
echo    3. Go to http://localhost:5000
echo    4. Login with any credentials
echo    5. Enter locations in Chicago, IL
echo.
echo ⚠️  Press Ctrl+C to stop the server
echo.
echo Starting in 3 seconds...
timeout /t 3 /nobreak >nul

REM Start Flask app
python app.py

REM If app crashes, show error message
if errorlevel 1 (
    echo.
    echo ========================================
    echo ❌ Application stopped with an error!
    echo ========================================
    echo.
    echo Common issues:
    echo 1. Port 5000 is already in use
    echo 2. Missing dependencies
    echo 3. Model or data file issues
    echo.
    echo Check the error messages above for details.
    echo.
)

pause