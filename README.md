# 🎬 Dialogue Localization Engine & Web App

A high-performance, full-stack AI pipeline that pinpoints the **exact video frame** where a specific spoken dialogue line begins. Built for robustness, speed, and visual clarity.

---

## 📋 Problem Statement

Given a publicly accessible video URL (e.g., [https://ok.ru/video/248244667877](https://ok.ru/video/248244667877)) and a target spoken line (e.g., **"My mind rebels at stagnation"**):
1. Locate the **exact video frame** and **precise timestamp** where the dialogue starts.
2. Extract the dialogue text from the audio to verify the transcription.
3. Export the target frame as a high-fidelity image file.
4. Provide a robust, automated solution that handles variations in video quality, frame rates, and speaker accents without requiring manual human scanning.

---

## 🛠️ Step-by-Step Solution

The application processes requests through a deterministic, modular pipeline:

1. **Pre-Network URL Cache Check:** Parses the video ID directly from the URL. If the MP4 is already downloaded in `./output/video`, it skips all network requests and queries entirely.
2. **Video Ingestion & Audio Extraction:** If a cache miss occurs, `yt-dlp` downloads the video container (using browser emulation and TLS 1.2 flags to bypass SSL handshake blocks). `ffmpeg` then extracts a mono 16 kHz WAV file.
3. **Speech Activity Segmentation (VAD):** Runs a Silero VAD (ONNX) model to isolate voice intervals from background noise and music.
4. **VAD Merging & Duration Filtering:** Adjacent speech fragments separated by $\le$ 1.5 seconds of silence are merged. Segments shorter than 2.0s are automatically skipped, as a 7-word phrase cannot physically be spoken in less time.
5. **ASR Pass 1: SCAN (Coarse Match):** Runs the lightweight `tiny` model with greedy decoding (`beam_size=1`) and no word-level timestamps to scan the full video. A fuzzy match locates the coarse onset window.
6. **ASR Pass 2: REFINE (Precise Transcription):** Unloads the `tiny` model from RAM and loads the `small` model. Transcribes a tight $\pm$30-second window (60s total) around the coarse onset with beam search (`beam_size=5`) and word timestamps.
7. **Forced Phoneme Alignment:** Executes a **WhisperX aligner** (wav2vec2 CTC model) on the 60s window to align phonemes to the audio waveform, recovering any words Whisper dropped via gap-filling.
8. **PTS Frame Containment Mapping:** Reads precise frame presentation timestamps (PTS) around the onset using `ffprobe`. Matches the precise audio onset to the exact enclosing frame interval.
9. **Confidence & Extraction:** Computes a composite confidence score (combining text match, ASR probability, and VAD corroboration). Extracts the target frame image via `ffmpeg`.

---

## 🎨 Application Navigation

The system can be operated in two ways:

### 1. Interactive Web Dashboard (Recommended)
Launch the full-stack application locally:
```powershell
# Activate venv & run the FastAPI server
.venv\Scripts\python.exe web_server.py --port 8000
```
Then open your browser to **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** to access a dashboard featuring:
- **Execute Pipeline Form:** Input any URL and target phrase.
- **Visual Progress Checklist:** A live stage-by-stage pipeline checklist that displays checkmarks, execution times, and active loading spinners.
- **Log Terminal:** Streams backend prints live with custom colors for scanning, refining, and error logs.
- **Interactive Results Panel:** Renders the extracted frame, composite confidence breakdown gauges, and exact timestamps.
- **Scan History Sidebar:** Instantly reload and inspect previous localization runs.

### 2. Command Line Interface (CLI)
Run a quick, direct scan from your console:
```powershell
.venv\Scripts\python.exe cli.py --url "https://ok.ru/video/248244667877" --target "My mind rebels at stagnation"
```

---

## 💡 Ambiguities & Technical Challenges Solved

- **The CPU Resource Bottleneck:** Whisper models are heavy. Running a full transcription with `medium/small` models and WhisperX alignment on a full 50-minute video on CPU takes **upwards of 50 minutes**. We solved this by implementing the **Two-Pass SCAN/REFINE hybrid architecture**, reducing processing time to **~2–4 minutes** (a **10× speedup**).
- **Network Handshake Resets (Odnoklassniki/OK.ru):** Foreign video hosts frequently block metadata probes or trigger SSL resets (`ConnectionResetError: 10054`) on Python 3.14. We solved this by implementing a **Pre-Network Cache Check** that resolves video IDs from the URL and checks the local directory before connecting.
- **Whisper Timestamp Drift:** Raw Whisper timestamps can drift by 500ms+. We solved this by running a **wav2vec2 forced aligner** inside the 1-minute refine window, aligning character boundaries to ~20–40ms, and mapping the onset to actual frame presentation timestamps (`ffprobe` PTS) using interval containment.
- **ASR calling overhead:** Raw VAD yields hundreds of short segments (e.g. 724 segments). Calling Whisper on each segment introduces massive startup overhead. We solved this by introducing **gap merging (1.5s threshold)** and **duration filtering**, collapsing the segment count to 74 and skipping noise entirely.

---

## 🧬 Model Inner Workings

1. **Silero VAD (ONNX):** A deep learning recurrent neural network (RNN) trained on multilingual corpora to classify audio frames as speech or non-speech.
2. **faster-whisper (CTranslate2):** A fast implementation of OpenAI's Whisper model. It uses a Transformer-based encoder-decoder. The encoder maps audio spectrograms to features, and the decoder autoregressively predicts character tokens.
3. **WhisperX Aligner (wav2vec2 CTC):** A Connectionist Temporal Classification (CTC) phoneme recognizer. It aligns character segments directly to the audio waveform by mapping phoneme probabilities to time frames.

---

## 📈 Performance & Iterative Improvements

The following table displays how performance progressed through different design iterations on a standard CPU:

| Iteration | Pipeline Architecture | Execution Time | Accuracy / Confidence |
| :--- | :--- | :--- | :--- |
| **v1.0 (Baseline)** | Single-pass ASR (`medium` model) + Full-file WhisperX forced alignment. | ~3,108 seconds (51 mins) | 0.946 (HIGH) |
| **v1.1 (VAD Merge)** | Merged VAD speech gaps ($\le$ 1.5s) to reduce Whisper call overhead. | ~1,240 seconds (20 mins) | 0.946 (HIGH) |
| **v2.0 (Two-Pass SCAN/REFINE)** | Two-Pass Hybrid ASR (`tiny` scan + `small` refine) + restricted window alignment. | **311 seconds** (YouTube download)<br>**145 seconds** (OK.ru cached) | **0.946 (HIGH)** |
| **v2.1 (Local Caching)** | Pre-network cache check utilizing video ID extraction. | **0.01 seconds** (Ingestion Phase) | 0.946 (HIGH) |

---

## 🤖 AI Co-Pilot & Engineering Acknowledgement

This system was engineered in collaboration with **Antigravity** (Google DeepMind's agentic coding assistant). 

- **LLM / Co-Pilot Prompts:** The exact prompts, design discussions, and research queries used during development are documented in [`engineering_prompts.txt`](file:///C:/Users/Akash/Documents/Draft%20Q1/engineering_prompts.txt) in this repository, satisfying the evaluation criteria.
- **AI Role:** Assisted in implementing the two-pass transcription API refactor, drafting the sequential queue worker thread in the backend, designing the custom log-capturing hander, and debugging Windows console Unicode encoding errors.
- **Developer Role:** Guided the architecture, defined structural constraints, and verified execution.

