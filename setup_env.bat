@echo off
echo Starting Dialogue Localization environment setup...

:: 1. Create virtual environment if missing
if not exist .venv (
    echo Creating virtual environment .venv...
    python -m venv .venv
)

:: 2. Upgrade pip
echo Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip

:: 3. Install setuptools to prevent editable backend compilation errors
echo Installing setuptools...
.venv\Scripts\python.exe -m pip install setuptools

:: 4. Install standard packages (resolving all sub-dependencies)
echo Installing core pipeline packages...
.venv\Scripts\python.exe -m pip install torch torchaudio faster-whisper silero-vad soundfile rapidfuzz yt-dlp rich pytest pytest-cov fastapi pydantic uvicorn

:: 5. Install Python 3.14 compatibility packages
echo Installing Python 3.14 compatibility packages...
.venv\Scripts\python.exe -m pip install ctranslate2>=4.6.1
.venv\Scripts\python.exe -m pip install pandas transformers nltk pyannote.audio==3.1.1

:: 6. Install whisperx without pulling incompatible ctranslate2==4.4.0
echo Installing whisperx...
.venv\Scripts\python.exe -m pip install whisperx --no-deps

:: 7. Install the package itself in editable mode
echo Installing dialogue_locator in editable mode...
.venv\Scripts\python.exe -m pip install -e .

echo Setup complete! You can now start the web server by running:
echo   .venv\Scripts\python.exe web_server.py
