#!/bin/bash
# Setup environment script for Linux/macOS
echo "Starting Dialogue Localization environment setup..."

# 1. Create virtual environment if missing
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment .venv..."
    python3 -m venv .venv
fi

# 2. Upgrade pip
echo "Upgrading pip..."
.venv/bin/python -m pip install --upgrade pip

# 3. Install setuptools to prevent editable backend compilation errors
echo "Installing setuptools..."
.venv/bin/python -m pip install setuptools

# 4. Install standard packages (resolving all sub-dependencies)
echo "Installing core pipeline packages..."
.venv/bin/python -m pip install torch torchaudio faster-whisper silero-vad soundfile rapidfuzz yt-dlp rich pytest pytest-cov fastapi pydantic uvicorn

# 5. Install Python 3.14 compatible CTranslate2 and dependencies
echo "Installing Python 3.14 compatibility packages..."
.venv/bin/python -m pip install "ctranslate2>=4.6.1"
.venv/bin/python -m pip install pandas transformers nltk pyannote.audio==3.1.1

# 6. Install whisperx without pulling incompatible ctranslate2==4.4.0
echo "Installing whisperx..."
.venv/bin/python -m pip install whisperx --no-deps

# 7. Install the package in editable mode
echo "Installing dialogue_locator in editable mode..."
.venv/bin/python -m pip install -e .

echo "Setup complete! You can now start the web server by running:"
echo "  .venv/bin/python web_server.py"
