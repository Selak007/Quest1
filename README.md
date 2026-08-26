# Spoken Dialogue Frame Localization Application

## Introduction

This application is a full-stack, high-performance software system designed to localize the exact video frame where a specific spoken dialogue line begins. It processes media URLs (such as YouTube, OK.ru, or other platforms supported by yt-dlp) and maps the speech onset directly to a precise video frame index and timestamp.

The application is structured as a Python-based processing pipeline, exposed via both a Command Line Interface (CLI) and an interactive FastAPI web dashboard. To ensure CPU efficiency, it uses a hybrid two-pass Automatic Speech Recognition (ASR) architecture and restricted wav2vec2 Connectionist Temporal Classification (CTC) forced alignment.

Detailed analysis of the problem statement, architecture pipeline design, ASR concepts, working examples, accuracy metrics, results, and model tradeoffs is documented in the design document:

*   **Design & Approach Description:** [approach.md](approach.md)
*   **Mermaid Architecture Source Code:** [approach.mmd](approach.mmd)

---

## Web Application Packaging

The application is packaged as a lightweight, full-stack service to make it easy to run and test:

1. **FastAPI Backend (web_server.py):**
   * Exposes endpoints to submit tasks (`POST /api/locate`), query status (`GET /api/tasks/{id}`), and list recent task histories.
   * Implements a **background task queue thread** that processes localization jobs sequentially. This prevents the server from attempting to run multiple resource-heavy Whisper models in parallel, avoiding CPU exhaustion.
   * Uses Server-Sent Events (SSE) via a log-streaming handler to broadcast console logs directly to the frontend.
   * Employs local JSON database persistence (`output/tasks.json`) to retain task logs and results across server restarts.
2. **Vanilla Frontend (static/):**
   * Designed with a dark-themed, glassmorphic UI using standard HTML5, CSS3, and JavaScript.
   * Includes a live progress checklist showing active loading spinners, completion checkmarks, and individual stage durations.
   * Displays confidence breakdown gauges (Text Match, ASR Quality, VAD Agreement) and a built-in frame image viewer.

---

## Application Setup and Execution Guide

This section outlines how to configure, start, and run the pipeline locally.

### 1. Prerequisites and Installation
Ensure Python 3.14 and ffmpeg/ffprobe are installed and available on your path. Then, configure the local environment:
```powershell
# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Install requirements. For Python 3.14, whisperx must be installed without 
# its strict dependencies to bypass the incompatible ctranslate2==4.4.0 check:
pip install -r requirements.txt --no-deps
pip install ctranslate2>=4.6.1
pip install pandas transformers nltk faster-whisper silero-vad soundfile rapidfuzz yt-dlp rich pytest pytest-cov
pip install whisperx --no-deps

# Install the package itself in editable mode
pip install -e .
```

### 2. Launching the Web Server
To start the FastAPI server and access the interactive web interface:
```powershell
python web_server.py --port 8000
```
Once the server starts, open your web browser and navigate to:
```
http://127.0.0.1:8000/
```
In the browser, you can submit URLs, watch the live log console stream, view the active step-by-step checklist, and download the extracted frame image.

### 3. Running via Command Line (CLI)
You can invoke the pipeline directly via terminal command:
```powershell
python cli.py --url "https://ok.ru/video/248244667877" --target "My mind rebels at stagnation"
```
Available CLI options include:
*   `--url, -u`: The media URL (YouTube, OK.ru, etc.).
*   `--target, -t`: The dialogue text to localize.
*   `--output-dir, -o`: Custom directory for outputs (default: `./output`).
*   `--no-frame`: Skip frame extraction for a faster raw text scan.
*   `--verbose, -v`: Enable debug level prints.

### 4. Running Unit Tests
Validate the pipeline modules (fully mocked inputs, no model downloads or internet needed):
```powershell
# Run the entire test suite
pytest tests/ -v

# Run with test coverage metrics
pytest tests/ --cov=dialogue_locator --cov-report=term-missing
```
