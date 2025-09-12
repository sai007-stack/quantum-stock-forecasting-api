#!/usr/bin/env python3
"""
Run script for Quantum Stock Market Forecasting application.
This script handles installation verification and application startup.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def run_tests():
    """Run installation tests."""
    print("🧪 Running installation tests...")
    try:
        result = subprocess.run([sys.executable, "test_installation.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def start_application():
    """Start the Streamlit application."""
    print("🚀 Starting Quantum Stock Market Forecasting application...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except Exception as e:
        print(f"❌ Failed to start application: {e}")

def main():
    """Main function."""
    print("🔮 Quantum Stock Market Forecasting")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found!")
        sys.exit(1)
    
    # Check if app.py exists
    if not Path("app.py").exists():
        print("❌ app.py not found!")
        sys.exit(1)
    
    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Install dependencies and run tests")
    print("2. Run tests only")
    print("3. Start application directly")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        if install_dependencies():
            if run_tests():
                start_application()
    elif choice == "2":
        if run_tests():
            start_application()
    elif choice == "3":
        start_application()
    elif choice == "4":
        print("👋 Goodbye!")
        sys.exit(0)
    else:
        print("❌ Invalid choice!")
        sys.exit(1)

if __name__ == "__main__":
    main()
