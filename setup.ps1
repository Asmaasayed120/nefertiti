# setup.ps1 - Nefertiti AI FastAPI Setup Script (Windows)

Write-Host "Setting up Nefertiti AI FastAPI Server..."

# 1. Check for Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Error "Python not found. Please install Python 3.10+ before continuing."
    exit 1
}

# 2. Create virtual environment
Write-Host "Creating virtual environment..."
python -m venv venv

# 3. Activate virtual environment
Write-Host "Activating virtual environment..."
. .\venv\Scripts\Activate.ps1

# 4. Upgrade pip
Write-Host "Upgrading pip..."
pip install --upgrade pip

# 5. Install PyTorch (CPU version)
Write-Host "Installing PyTorch CPU version..."
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# 6. Install requirements
Write-Host "Installing requirements..."
pip install -r requirements.txt

# 7. Clone Wav2Lip
if (-not (Test-Path "Wav2Lip")) {
    Write-Host "Cloning Wav2Lip..."
    git clone https://github.com/Rudrabha/Wav2Lip.git
} else {
    Write-Host "Wav2Lip folder already exists, skipping clone..."
}

# 8. Download model
if (-not (Test-Path "wav2lip_gan.pth")) {
    Write-Host "Downloading wav2lip_gan.pth model..."
    Invoke-WebRequest -Uri "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth" -OutFile "wav2lip_gan.pth"
} else {
    Write-Host "Model already exists, skipping download..."
}

# 9. Create necessary directories
Write-Host "Creating folders..."
New-Item -ItemType Directory -Path temp_files -Force | Out-Null
New-Item -ItemType Directory -Path logs -Force | Out-Null

Write-Host "`n✅ Setup completed!"
Write-Host "To run the app:"
Write-Host "1. .\venv\Scripts\Activate.ps1"
Write-Host "2. uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
Write-Host "`nAPI Docs: http://localhost:8000/docs"
