"""
SafeRoute Setup Checker
This script verifies your setup is correct before running the application.
"""

import os
import sys
from pathlib import Path

def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def check_item(name, condition, required=True):
    """Check if a condition is met and print result"""
    status = "✅" if condition else ("❌" if required else "⚠️")
    print(f"{status} {name}")
    return condition

def main():
    print_header("SafeRoute Setup Verification")
    
    all_checks_passed = True
    
    # 1. Check Python version
    print("\n📍 Checking Python Version...")
    python_version = sys.version_info
    version_ok = python_version.major == 3 and python_version.minor >= 8
    check_item(
        f"Python {python_version.major}.{python_version.minor}.{python_version.micro}",
        version_ok
    )
    if not version_ok:
        print("   → Python 3.8 or higher is required")
        all_checks_passed = False
    
    # 2. Check required files
    print("\n📁 Checking Required Files...")
    required_files = {
        'app.py': True,
        'route_predictor.py': True,
        'cleaned_chicago_crime_data.csv': True,
        'random_forest_model.pkl': True,
        'requirements_web.txt': True,
        'templates/index.html': True,
        'templates/map.html': True,
    }
    
    for file, required in required_files.items():
        exists = Path(file).exists()
        if not check_item(file, exists, required):
            if required:
                all_checks_passed = False
    
    # 3. Check optional files
    print("\n📋 Checking Optional Files...")
    optional_files = [
        'features_for_model_final.csv',
        'feature.py',
        'train_random_forest.py',
        'start.bat'
    ]
    
    for file in optional_files:
        check_item(file, Path(file).exists(), required=False)
    
    # 4. Check directories
    print("\n📂 Checking Directories...")
    required_dirs = ['templates', 'static']
    for dir_name in required_dirs:
        exists = Path(dir_name).exists()
        if not check_item(f"{dir_name}/", exists):
            all_checks_passed = False
            print(f"   → Create with: mkdir {dir_name}")
    
    # 5. Check Python packages
    print("\n📦 Checking Python Packages...")
    required_packages = [
        'flask',
        'flask_cors',
        'osmnx',
        'networkx',
        'folium',
        'joblib',
        'pandas',
        'numpy',
        'geopy',
        'scipy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            check_item(package, True)
        except ImportError:
            check_item(package, False)
            missing_packages.append(package)
            all_checks_passed = False
    
    if missing_packages:
        print("\n   → Install missing packages with:")
        print("   pip install -r requirements_web.txt")
    
    # 6. Check model file size
    print("\n🤖 Checking Model File...")
    model_path = Path('random_forest_model.pkl')
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        check_item(f"Model size: {size_mb:.2f} MB", size_mb > 0.1)
        if size_mb < 0.1:
            print("   → Model file seems too small, retrain with:")
            print("   python train_random_forest.py")
    
    # 7. Check data file
    print("\n📊 Checking Crime Data File...")
    data_path = Path('cleaned_chicago_crime_data.csv')
    if data_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(data_path, nrows=5)
            required_cols = {'Latitude', 'Longitude'}
            has_cols = required_cols.issubset(df.columns)
            check_item(
                f"Data has required columns: {', '.join(required_cols)}", 
                has_cols
            )
            if not has_cols:
                print(f"   → Found columns: {', '.join(df.columns)}")
                all_checks_passed = False
            
            size_mb = data_path.stat().st_size / (1024 * 1024)
            check_item(f"Data size: {size_mb:.2f} MB", size_mb > 1)
        except Exception as e:
            check_item("Data file readable", False)
            print(f"   → Error: {e}")
            all_checks_passed = False
    
    # 8. Check port availability
    print("\n🌐 Checking Port Availability...")
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 5000))
        port_free = result != 0
        sock.close()
        
        if not port_free:
            check_item("Port 5000 available", False, required=False)
            print("   → Port 5000 is in use. Options:")
            print("   1. Stop the process using port 5000")
            print("   2. Change port in app.py to 5001")
        else:
            check_item("Port 5000 available", True)
    except Exception as e:
        check_item(f"Port check (skipped: {e})", True, required=False)
    
    # Final summary
    print_header("Summary")
    
    if all_checks_passed:
        print("\n🎉 All checks passed! You're ready to run SafeRoute.")
        print("\n📝 To start the application:")
        print("   • Windows: Run start.bat")
        print("   • Linux/Mac: python app.py")
        print("\n🌐 Then open: http://localhost:5000")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\n🔧 Common fixes:")
        print("   1. Install dependencies: pip install -r requirements_web.txt")
        print("   2. Create templates folder: mkdir templates")
        print("   3. Move HTML files to templates/")
        print("   4. Train model: python train_random_forest.py")
        print("   5. Ensure data file exists: cleaned_chicago_crime_data.csv")
    
    print("\n" + "=" * 60)
    return 0 if all_checks_passed else 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ Check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)