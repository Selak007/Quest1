"""
FastAPI web server for Dialogue Frame Localization.

Provides:
- Asynchronous task processing (sequential queue to prevent CPU overload).
- SSE endpoint for real-time progress log streaming.
- Static serving of frontend assets (HTML, CSS, JS) and output frame images.
"""
from __future__ import annotations

import uuid
import queue
import threading
import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse

from dialogue_locator.pipeline import run_pipeline, LocalizationResult
from dialogue_locator import config as cfg

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
root_logger = logging.getLogger()

# In-memory database of tasks with JSON persistence
TASKS: Dict[str, Dict[str, Any]] = {}
current_task_id: List[Optional[str]] = [None]  # List wrapper for thread-safe updates

TASKS_FILE = Path("./output/tasks.json")

def save_tasks_to_disk():
    try:
        TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TASKS_FILE, "w", encoding="utf-8") as f:
            json.dump(TASKS, f, ensure_ascii=False, indent=2)
    except Exception as err:
        logging.error(f"[Server] Failed to save tasks to disk: {err}")

def load_tasks_from_disk():
    global TASKS
    if TASKS_FILE.exists():
        try:
            with open(TASKS_FILE, "r", encoding="utf-8") as f:
                TASKS = json.load(f)
            logging.info(f"[Server] Loaded {len(TASKS)} tasks from disk.")
        except Exception as err:
            logging.error(f"[Server] Failed to load tasks from disk: {err}")
            TASKS = {}
    else:
        TASKS = {}


# Thread-safe Task log capturing handler
class TaskLogHandler(logging.Handler):
    def emit(self, record):
        try:
            task_id = current_task_id[0]
            if task_id and task_id in TASKS:
                log_entry = self.format(record)
                TASKS[task_id]["logs"].append(log_entry)
                
                # Check for progress percentage in logs (e.g. "31.6% done")
                # Both [SCAN] and [REFINE] phases update percentage.
                if "done" in log_entry or "complete" in log_entry:
                    if "%" in log_entry:
                        try:
                            # Extract percentage float
                            parts = log_entry.split("%")
                            subparts = parts[0].split(",")
                            if len(subparts) > 1:
                                pct = float(subparts[-1].strip())
                            else:
                                pct = float(subparts[0].split("(")[-1].strip())
                            TASKS[task_id]["progress"] = pct
                        except Exception:
                            pass
        except Exception:
            pass

# Attach the handler to capture log outputs
log_handler = TaskLogHandler()
log_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%H:%M:%S"))
root_logger.addHandler(log_handler)


# ── Background Worker Queue ───────────────────────────────────────────────────
task_queue = queue.Queue()

def worker_thread_loop():
    """Background thread that runs pipeline tasks sequentially."""
    while True:
        try:
            task_id = task_queue.get()
            if task_id is None:
                break
            
            task = TASKS[task_id]
            task["status"] = "running"
            current_task_id[0] = task_id
            save_tasks_to_disk()
            
            logging.info(f"[Server] Starting Task {task_id}: {task['target']}")
            task["logs"].append(f"[Server] Task started at {datetime.now().strftime('%H:%M:%S')}")
            
            t0 = time.perf_counter()
            try:
                # Run the actual pipeline
                result: LocalizationResult = run_pipeline(
                    url=task["url"],
                    target=task["target"],
                    output_dir=Path("./output"),
                    extract_frame_image=True,
                    strict=False,
                )
                
                # Format response result dict
                task["result"] = {
                    "timestamp_s": result.timestamp_s,
                    "timestamp_fmt": result.timestamp_fmt,
                    "frame_number": result.frame_number,
                    "dialogue_text": result.dialogue_text,
                    "matched_text": result.matched_text,
                    "confidence": result.confidence,
                    "status": result.status,
                    "text_score": result.text_score,
                    "asr_quality": result.asr_quality,
                    "vad_agreement": result.vad_agreement,
                    "vad_transition_s": result.vad_transition_s,
                    # Format frame path for client URL consumption
                    "frame_image_url": f"/output/frames/{Path(result.frame_image_path).name}" if result.frame_image_path else None
                }
                task["status"] = "completed"
                task["progress"] = 100.0
                task["logs"].append(f"[Server] Task completed successfully in {time.perf_counter() - t0:.2f}s")
                
            except Exception as exc:
                logging.error(f"[Server] Task {task_id} failed: {exc}", exc_info=True)
                task["status"] = "failed"
                task["error"] = str(exc)
                task["logs"].append(f"[Server] Error: {exc}")
            
            finally:
                save_tasks_to_disk()
                current_task_id[0] = None
                task_queue.task_done()
                
        except Exception as queue_exc:
            logging.error(f"[Server] Worker thread exception: {queue_exc}")
            time.sleep(1.0)

# Start worker thread
worker_thread = threading.Thread(target=worker_thread_loop, daemon=True)
worker_thread.start()


# ── FastAPI App Configuration ──────────────────────────────────────────────────
app = FastAPI(
    title="Dialogue Localization API",
    description="Backend for Spoken Dialogue Exact Frame Localization",
    version="2.0.0"
)

# Enable CORS for local testing/dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Request Schema ───────────────────────────────────────────────────
class LocateRequest(BaseModel):
    url: str
    target: str


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/locate")
def create_locate_task(req: LocateRequest):
    """Submit a video URL and target phrase for frame localization."""
    if not req.url or not req.target:
        raise HTTPException(status_code=400, detail="URL and Target Phrase are required.")
    
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "task_id": task_id,
        "url": req.url,
        "target": req.target,
        "status": "pending",
        "progress": 0.0,
        "logs": [f"[Server] Task submitted at {datetime.now().strftime('%H:%M:%S')}"],
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat()
    }
    
    task_queue.put(task_id)
    save_tasks_to_disk()
    return {"task_id": task_id, "status": "pending"}


@app.get("/api/tasks")
def list_tasks():
    """List all recent tasks with their state summaries."""
    summary = []
    # Sort tasks by creation time (newest first)
    sorted_tasks = sorted(TASKS.values(), key=lambda t: t["created_at"], reverse=True)
    for t in sorted_tasks:
        summary.append({
            "task_id": t["task_id"],
            "url": t["url"],
            "target": t["target"],
            "status": t["status"],
            "progress": t["progress"],
            "created_at": t["created_at"],
            "has_result": t["result"] is not None
        })
    return summary


@app.get("/api/tasks/{task_id}")
def get_task_details(task_id: str):
    """Retrieve full status details for a given task ID."""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    return TASKS[task_id]


@app.get("/api/tasks/{task_id}/logs")
def stream_task_logs(task_id: str):
    """Server-Sent Events (SSE) log stream for real-time progress logging."""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")

    async def sse_log_generator():
        last_idx = 0
        while True:
            task = TASKS[task_id]
            logs = task["logs"]
            
            # Yield any new log lines since the last iteration
            if last_idx < len(logs):
                for idx in range(last_idx, len(logs)):
                    line = logs[idx].replace("\n", " ").strip()
                    yield f"data: {line}\n\n"
                last_idx = len(logs)
            
            # If the task finishes, send completion signal and close SSE
            if task["status"] in ("completed", "failed"):
                # Final flush
                if last_idx < len(logs):
                    for idx in range(last_idx, len(logs)):
                        line = logs[idx].replace("\n", " ").strip()
                        yield f"data: {line}\n\n"
                yield "data: [DONE]\n\n"
                break
                
            import asyncio
            await asyncio.sleep(0.3)

    return StreamingResponse(sse_log_generator(), media_type="text/event-stream")


# ── Static/Frontend Assets Serving ────────────────────────────────────────────

# Make sure output directories exist
Path("./output/frames").mkdir(parents=True, exist_ok=True)
Path("./static").mkdir(parents=True, exist_ok=True)

# Mount `./output` under `/output` to let the frontend resolve images directly
app.mount("/output", StaticFiles(directory="./output"), name="output")

@app.get("/")
def serve_home():
    """Serves the main HTML5 frontend."""
    index_path = Path("./static/index.html")
    if not index_path.exists():
        return {"message": "Server started successfully. Please create index.html in ./static directory."}
    return FileResponse(index_path)

# Mount remaining static assets (CSS, JS) under `/static`
app.mount("/static", StaticFiles(directory="./static"), name="static")


if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Parse CLI overrides for host/port if needed
    host = "127.0.0.1"
    port = 8000
    for idx, arg in enumerate(sys.argv):
        if arg == "--host" and idx + 1 < len(sys.argv):
            host = sys.argv[idx + 1]
        elif arg == "--port" and idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])
            
    load_tasks_from_disk()
    print(f"\nDialogue Localization Web Application Starting!")
    print(f"URL: http://{host}:{port}/\n")
    
    uvicorn.run(app, host=host, port=port, log_level="warning")
