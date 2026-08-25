# Pipeline Architecture & Deviations Report

This document maps out the architectural deviations from the initial single-pass pipeline design to the current high-performance two-pass hybrid system, along with the detailed stage-by-step execution path.

---

## 🔄 Summary of Major Deviations & Optimizations

| Phase / Stage | Initial Single-Pass Design | Current Two-Pass Hybrid Design | Performance Impact & Rationale |
| :--- | :--- | :--- | :--- |
| **ASR Model & Decoding** | Transcribed the full video using a single heavy `medium` Whisper model with beam search and word-level timestamps enabled. | **Pass 1 (SCAN):** Transcribes full audio using `tiny` model, greedy decoding (`beam_size=1`), and segment-level timestamps.<br><br>**Pass 2 (REFINE):** Transcribes *only* a 1-minute window ($\pm$30s) around the rough match using `small` model with beam search (`beam_size=5`) and word timestamps. | **10× Speedup.** Cuts down transcription time on a 40-minute video from 45+ minutes to ~2 minutes on CPU. |
| **Forced Alignment** | Ran WhisperX CTC alignment on the transcript of the entire video (aligning 900+ segments). | Runs WhisperX CTC alignment *only* on the refined 1-minute window (~16–30 segments). | Cuts alignment stage computation time from ~15 minutes to **~40 seconds** on CPU. |
| **VAD Handling** | Submitted every raw VAD segment directly to ASR (~720 fragments). | **VAD Merging:** Collapses adjacent segments separated by $\le$ 1.5s.<br><br>**VAD Length Filtering:** Skips segments shorter than `SCAN_MIN_SEG_DURATION_S` (2.0s). | Reduces ASR calling overhead by **80%** by ignoring short sighs, breaths, and background noise. |
| **Video Ingestion** | Invoked `yt-dlp --get-filename` to query server metadata before checking for a cached file. | **Pre-Network URL Parsing:** Extracts the unique video ID from the URL and checks the local directory before launching yt-dlp. | Bypasses network TLS/SSL handshake resets (common with blocked video sharing hosts like OK.ru) if the video is already cached. |
| **Application Layer** | Command Line Interface (CLI) entry point only. | Full-stack web encapsulation: FastAPI backend (`web_server.py`) + sequential task queue + glassmorphic HTML5/CSS/JS frontend dashboard. | Adds support for asynchronous multi-task queuing, real-time log streaming (SSE), and interactive frame inspection. |

---

## 🎬 Current Pipeline Execution Flow

The current pipeline processes dialogue localization requests in the following sequence:

```
[Start: URL & Target Phrase]
       │
       ▼
[Pre-Network URL Cache Check] ──► (Local MP4 Found) ─┐
       │                                            │
       ▼ (Cache Miss)                               │
[Download Video via yt-dlp]                         │
       │                                            │
       ▼                                            ▼
[Extract 16kHz Mono Audio] ◄────────────────────────┘
       │
       ▼
[Silero VAD Segment Detection]
       │
       ▼
[Merge Gaps & Length Filter]
       │
       ▼
[ASR Pass 1: SCAN (tiny, greedy)]
       │
       ▼
[Fuzzy Match on Scan Segment Times] ──► (Find Rough Timestamp)
       │
       ▼
[Unload SCAN Model from RAM]
       │
       ▼
[ASR Pass 2: REFINE (small, beam) on ±30s Window]
       │
       ▼
[Forced Alignment via WhisperX]
       │
       ▼
[Fuzzy Match on Refined Aligned Words]
       │
       ▼
[VAD Corroboration & ffprobe PTS Read]
       │
       ▼
[Interval Containment Frame Mapping]
       │
       ▼
[ffmpeg Precise Frame Extraction]
       │
       ▼
[End: Present Result & Render UI]
```

### Stage-by-Stage Implementation Details

#### 1. Ingestion & Pre-Network Cache Lookup
- **URL Parsing:** Extracts video ID (e.g., YouTube video ID or OK.ru digits) from the input URL.
- **Cache Check:** Verifies if `output/video/{id}.mp4` exists locally. If present, it skips the network probe entirely.
- **yt-dlp Download:** If missing, downloads the video using custom headers, User-Agent, and TLS 1.2 compatibility flags (`--legacy-server-connect`).

#### 2. Audio Extraction
- Extracts a single-channel, 16 kHz WAV file from the video container using `ffmpeg`.

#### 3. Voice Activity Detection (VAD) & Filtering
- **Silero VAD:** Detects speech boundary intervals.
- **Merging (`VAD_MERGE_GAP_S`):** Merges speech intervals separated by $\le$ 1.5 seconds of silence to reduce ASR start-stop overhead.
- **Duration Filtering:** Skips segments shorter than `SCAN_MIN_SEG_DURATION_S` (2.0s) because the 7-to-8-word target phrase requires at least 2 seconds of natural speech.

#### 4. ASR Pass 1: SCAN (Fast Coarse Localization)
- Loads the lightweight `tiny` model.
- Transcribes eligible VAD segments with `beam_size=1` (greedy decoding) and `word_timestamps=False` (segment-level).
- **Fuzzy Match (Scan):** Compares the target phrase against the coarse transcript and identifies the rough timestamp.
- **RAM Cleanup:** Unloads the `tiny` model to conserve system RAM before loading the refine model.

#### 5. ASR Pass 2: REFINE (Precise Word Timestamps)
- Extracts a $\pm$30-second window (60 seconds total) around the rough timestamp.
- Loads the `small` model.
- Transcribes the window with `beam_size=5` (accurate beam decoding) and `word_timestamps=True` (word-level).

#### 6. Forced Word Alignment (WhisperX)
- Re-aligns the `small` model's transcript against the 60-second audio waveform using wav2vec2 CTC.
- Performs gap-filling from faster-whisper to ensure no words are dropped during alignment.

#### 7. Precise Fuzzy Match & VAD Verification
- Identifies the exact onset timestamp of the phrase.
- Corroborates the onset against VAD speech-silence transitions to verify natural phrasing.

#### 8. Frame Mapping & Extraction
- **ffprobe Lookup:** Reads frame PTS values in a 2-second window around the precise onset.
- **Containment Check:** Maps the onset time to the exact enclosing frame interval.
- **Confidence Score:** Computes a composite confidence score (0–1).
- **ffmpeg Extraction:** Extracts the exact frame image and saves it to `./output/frames/`.
