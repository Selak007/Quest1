"""
Spoken Dialogue → Exact Video Frame
Architecture v2: WhisperX Forced-Alignment-Based Onset Detection
"""
from .pipeline import run_pipeline, LocalizationResult

__all__ = ["run_pipeline", "LocalizationResult"]
__version__ = "0.1.0"
