# Training Setup Script for Windows PowerShell
# This script sets up the training environment

Write-Host "=" -NoNewline
Write-Host ("=" * 59)
Write-Host "Epsilon AI Training Setup"
Write-Host ("=" * 60)

# Check Python
Write-Host "`n[1/5] Checking Python installation..."
try {
    $pythonVersion = py --version 2>&1
    Write-Host "  ✓ Found: $pythonVersion"
} catch {
    Write-Host "  ✗ Python not found. Please install Python 3.8+ from python.org"
    exit 1
}

# Create virtual environment
Write-Host "`n[2/5] Creating virtual environment..."
if (Test-Path ".venv") {
    Write-Host "  ✓ Virtual environment already exists"
} else {
    py -m venv .venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Virtual environment created"
    } else {
        Write-Host "  ✗ Failed to create virtual environment"
        exit 1
    }
}

# Activate virtual environment
Write-Host "`n[3/5] Activating virtual environment..."
& ".venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ⚠ Activation failed. You may need to run:"
    Write-Host "     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
    Write-Host "  Then manually activate: .venv\Scripts\Activate.ps1"
}

# Upgrade pip
Write-Host "`n[4/5] Upgrading pip..."
python -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ pip upgraded"
} else {
    Write-Host "  ⚠ pip upgrade failed, continuing anyway..."
}

# Install requirements
Write-Host "`n[5/5] Installing training dependencies..."
Write-Host "  This may take a few minutes (PyTorch is large)..."
python -m pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ All dependencies installed"
} else {
    Write-Host "  ✗ Installation failed. Check errors above."
    exit 1
}

# Verify installation
Write-Host "`n[VERIFY] Running setup verification..."
python check_setup.py

Write-Host "`n" + ("=" * 60)
Write-Host "Setup complete!"
Write-Host ("=" * 60)
Write-Host "`nNext steps:"
Write-Host "  1. Make sure .env file exists in project root with SUPABASE_URL and SUPABASE_SERVICE_KEY"
Write-Host "  2. Activate virtual environment: .venv\Scripts\Activate.ps1"
Write-Host "  3. Run: python check_setup.py"
Write-Host "  4. Start training pipeline (see SETUP.md for details)"

