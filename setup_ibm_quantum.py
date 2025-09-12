#!/usr/bin/env python3
"""
Setup script for IBM Quantum connection.
This will help you connect to real IBM Quantum hardware.
"""

import subprocess
import sys

def install_ibm_provider():
    """Install IBM Quantum provider."""
    print("📦 Installing IBM Quantum provider...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "qiskit-ibm-provider"])
        print("✅ IBM Quantum provider installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install IBM Quantum provider: {e}")
        return False

def test_connection():
    """Test IBM Quantum connection."""
    print("\n🧪 Testing IBM Quantum connection...")
    try:
        from qiskit_ibm_provider import IBMProvider
        print("✅ IBM Quantum provider imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ IBM Quantum provider not available: {e}")
        return False

def main():
    """Main setup function."""
    print("🔮 IBM QUANTUM SETUP")
    print("=" * 30)
    
    # Install provider
    if install_ibm_provider():
        # Test connection
        if test_connection():
            print("\n🎉 Setup complete!")
            print("You can now run: python ibm_quantum_connection.py")
            print("Make sure you have your IBM Quantum API token ready!")
        else:
            print("\n❌ Setup failed - provider not available")
    else:
        print("\n❌ Setup failed - installation error")

if __name__ == "__main__":
    main()
