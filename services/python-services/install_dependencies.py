#!/usr/bin/env python3
"""
Epsilon AI - Python Dependencies Installer
© 2025 Neural Operation's & Holding's LLC. All rights reserved.

This script installs Python dependencies for Epsilon AI services
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_pip():
    """Check if pip is available"""
    try:
        subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                      check=True, capture_output=True)
        print("pip is available")
        return True
    except subprocess.CalledProcessError:
        print("Error: pip is not available")
        return False

def install_requirements():
    """Install requirements from requirements.txt"""
    requirements_file = Path(__file__).parent / 'requirements.txt'
    
    if not requirements_file.exists():
        print("Error: requirements.txt not found")
        return False
    
    try:
        print("Installing Python dependencies...")
        print("This may take a few minutes...")
        
        # Install requirements
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
        ], check=True, capture_output=True, text=True)
        
        print("All dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to install dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False

def install_optional_dependencies():
    """Install optional dependencies for enhanced functionality"""
    optional_packages = [
        'transformers',
        'torch',
        'sentence-transformers'
    ]
    
    print("\nInstalling optional dependencies for enhanced AI capabilities...")
    
    for package in optional_packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', package
            ], check=True, capture_output=True)
            print(f"{package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to install {package} (optional)")

def verify_installation():
    """Verify that key packages are installed"""
    key_packages = [
        'fastapi',
        'uvicorn',
        # 'aiohttp',  # Removed - not needed
        # 'numpy',  # Removed for compatibility
        # 'pandas'  # Removed for compatibility
    ]
    
    print("\nVerifying installation...")
    
    for package in key_packages:
        try:
            __import__(package)
            print(f"{package} is available")
        except ImportError:
            print(f"Error: {package} is not available")

def main():
    """Main installation function"""
    print("Epsilon AI Python Dependencies Installer")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check pip
    if not check_pip():
        print("Please install pip first")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("Error: Installation failed")
        sys.exit(1)
    
    # Install optional dependencies
    install_optional_dependencies()
    
    # Verify installation
    verify_installation()
    
    print("\nInstallation complete!")
    print("You can now start the Python services with:")
    print("  python start_services.py")
    print("  or")
    print("  npm run python-start")

if __name__ == "__main__":
    main()
