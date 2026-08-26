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

# 3. Install setuptools to prevent editable backend compilation errors
Write-Output "Installing setuptools..."
& .venv\Scripts\python.exe -m pip install setuptools

# 4. Install standard packages (resolving all sub-dependencies)
Write-Output "Installing core pipeline packages..."
& .venv\Scripts\python.exe -m pip install torch torchaudio faster-whisper silero-vad soundfile rapidfuzz yt-dlp rich fastapi pydantic uvicorn

# 5. Install Python 3.14 compatibility packages
Write-Output "Installing Python 3.14 compatibility packages..."
& .venv\Scripts\python.exe -m pip install ctranslate2>=4.6.1
& .venv\Scripts\python.exe -m pip install pandas transformers nltk pyannote.audio==3.1.1

# 6. Install whisperx without pulling incompatible ctranslate2==4.4.0
Write-Output "Installing whisperx..."
& .venv\Scripts\python.exe -m pip install whisperx --no-deps

# 7. Install the package itself in editable mode
Write-Output "Installing dialogue_locator in editable mode..."
& .venv\Scripts\python.exe -m pip install -e .

# 8. Copy validation video to output cache for offline out-of-the-box processing
Write-Output "Copying validation video to local cache..."
if (-not (Test-Path "output\video")) {
    New-Item -ItemType Directory -Force -Path "output\video" | Out-Null
}
if (Test-Path "docs\video\248244667877.mp4") {
    Copy-Item -Path "docs\video\248244667877.mp4" -Destination "output\video\248244667877.mp4" -Force
}

Write-Output "Setup complete! You can now start the web server by running:"
Write-Output "  .venv\Scripts\python.exe web_server.py"
