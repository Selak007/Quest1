# Setup environment script for Windows PowerShell
Write-Output "Starting Dialogue Localization environment setup..."

# 1. Create virtual environment if missing
if (-not (Test-Path ".venv")) {
    Write-Output "Creating virtual environment .venv..."
    python -m venv .venv
}

# 2. Upgrade pip
Write-Output "Upgrading pip..."
& .venv\Scripts\python.exe -m pip install --upgrade pip

# 3. Install dependencies with Python 3.14 compatibility workaround
Write-Output "Installing packages..."
& .venv\Scripts\python.exe -m pip install -r requirements.txt --no-deps
& .venv\Scripts\python.exe -m pip install ctranslate2>=4.6.1
& .venv\Scripts\python.exe -m pip install pandas transformers nltk faster-whisper silero-vad soundfile rapidfuzz yt-dlp rich pytest pytest-cov
& .venv\Scripts\python.exe -m pip install whisperx --no-deps

# 4. Install the package in editable mode
Write-Output "Installing dialogue_locator in editable mode..."
& .venv\Scripts\python.exe -m pip install -e .

Write-Output "Setup complete! You can now start the web server by running:"
Write-Output "  .venv\Scripts\python.exe web_server.py"
