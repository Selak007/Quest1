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

# 3. Install dependencies with Python 3.14 compatibility workaround
echo "Installing packages..."
.venv/bin/python -m pip install -r requirements.txt --no-deps
.venv/bin/python -m pip install "ctranslate2>=4.6.1"
.venv/bin/python -m pip install pandas transformers nltk faster-whisper silero-vad soundfile rapidfuzz yt-dlp rich pytest pytest-cov
.venv/bin/python -m pip install whisperx --no-deps

# 4. Install the package in editable mode
echo "Installing dialogue_locator in editable mode..."
.venv/bin/python -m pip install -e .

echo "Setup complete! You can now start the web server by running:"
echo "  .venv/bin/python web_server.py"
