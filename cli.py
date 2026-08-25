"""
CLI entry-point for the Spoken Dialogue Frame Localization tool.

Usage:
    python cli.py --url <VIDEO_URL> --target "My mind rebels at stagnation"

Full help:
    python cli.py --help
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
# with box-drawing and arrow characters used by rich.
console = Console(file=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") if hasattr(sys.stdout, "buffer") else sys.stdout)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=Console(
                    stderr=True,
                    file=io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace") if hasattr(sys.stderr, "buffer") else sys.stderr,
                ),
                rich_tracebacks=True,
                show_path=False,
            )
        ],
    )
    # Suppress noisy third-party loggers unless verbose
    if not verbose:
        for noisy in ["torch", "urllib3", "filelock", "numba", "httpx"]:
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dialogue-locator",
        description=(
            "🎬  Find the exact video frame where a spoken line begins.\n"
            "Architecture v2 — WhisperX forced alignment + interval-containment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--url", "-u",
        required=True,
        metavar="URL",
        help="Video URL (YouTube, ok.ru, or any yt-dlp-supported site).",
    )
    parser.add_argument(
        "--target", "-t",
        required=True,
        metavar="PHRASE",
        help='The spoken dialogue line to locate. e.g. "My mind rebels at stagnation"',
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./output",
        metavar="DIR",
        help="Directory for downloaded video, audio, and frame files. Default: ./output",
    )
    parser.add_argument(
        "--model", "-m",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size. Default: medium",
    )
    parser.add_argument(
        "--language", "-l",
        default="en",
        metavar="LANG",
        help="Audio language code. Default: en",
    )
    parser.add_argument(
        "--no-frame",
        action="store_true",
        help="Skip extracting the frame image (faster).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 if confidence is below the 'low' threshold.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def _print_result(result) -> None:
    """Render the LocalizationResult as a rich table."""

    # ── Status colour ─────────────────────────────────────────────────────────
    status_colours = {
        "high": "bright_green",
        "medium": "yellow",
        "low": "orange3",
        "best_effort": "red",
    }
    status_colour = status_colours.get(result.status, "white")

    console.print()
    console.rule("[bold cyan]🎬  Dialogue Localization Result[/]")
    console.print()

    # ── Main results table ────────────────────────────────────────────────────
    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    table.add_column("Field", style="bold cyan", width=22)
    table.add_column("Value", style="white")

    table.add_row("Timestamp", f"[bold white]{result.timestamp_fmt}[/]  ({result.timestamp_s:.3f}s)")
    table.add_row("Frame Number", f"[bold white]{result.frame_number:,}[/]")
    table.add_row("Target Phrase", f"[italic]{result.dialogue_text}[/]")
    table.add_row("ASR Match", f"[italic dim]{result.matched_text}[/]")
    table.add_row(
        "Confidence",
        f"[{status_colour}]{result.confidence:.3f}  ({result.status.upper()})[/]",
    )
    table.add_row(
        "Frame Window",
        f"{result.pts_window_start:.3f}s -> {result.pts_window_end:.3f}s",
    )

    console.print(table)
    console.print()

    # ── Confidence breakdown ──────────────────────────────────────────────────
    breakdown = Table(
        title="Confidence Breakdown",
        box=box.SIMPLE_HEAD,
        show_header=True,
        padding=(0, 2),
    )
    breakdown.add_column("Signal", style="bold")
    breakdown.add_column("Score", justify="right")
    breakdown.add_column("Weight", justify="right", style="dim")
    breakdown.add_row("Text Match",    f"{result.text_score:.3f}", "×0.50")
    breakdown.add_row("ASR Quality",   f"{result.asr_quality:.3f}", "×0.30")
    breakdown.add_row("VAD Agreement", f"{result.vad_agreement:.3f}", "×0.20")
    breakdown.add_row(
        "[bold]Composite[/]",
        f"[bold {status_colour}]{result.confidence:.3f}[/]",
        "",
    )

    if result.vad_transition_s is not None:
        breakdown.add_row(
            "  VAD transition at",
            f"{result.vad_transition_s:.3f}s",
            "",
        )

    console.print(breakdown)

    # ── File paths ────────────────────────────────────────────────────────────
    if result.frame_image_path:
        console.print()
        console.print(
            Panel(
                f"[bold]Frame image:[/]  {result.frame_image_path}",
                title="📁 Output Files",
                border_style="cyan",
            )
        )

    # ── Warning for low confidence ────────────────────────────────────────────
    if result.status in ("low", "best_effort"):
        console.print()
        console.print(
            Panel(
                f"[yellow]⚠  Confidence is [bold]{result.status.upper()}[/bold]. "
                f"The result may be unreliable.\n"
                f"   Tips: try a larger Whisper model (--model large-v2), "
                f"check the audio quality, or lower --fuzzy-cutoff.",
                border_style="yellow",
            )
        )
    console.print()


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    # Apply CLI overrides to config
    from dialogue_locator import config as cfg
    cfg.WHISPER_MODEL = args.model
    cfg.WHISPER_LANGUAGE = args.language

    console.print(
        Panel(
            f"[bold cyan]Spoken Dialogue → Exact Frame[/]\n"
            f"[dim]URL:   [/]{args.url}\n"
            f"[dim]Target:[/] [italic]{args.target}[/]",
            border_style="cyan",
        )
    )

    try:
        from dialogue_locator.pipeline import run_pipeline, LowConfidenceError

        result = run_pipeline(
            url=args.url,
            target=args.target,
            output_dir=Path(args.output_dir),
            extract_frame_image=not args.no_frame,
            strict=args.strict,
        )
        _print_result(result)
        return 0

    except ValueError as exc:
        console.print(f"\n[bold red]✗ Phrase not found:[/] {exc}")
        return 1

    except Exception as exc:
        if args.verbose:
            console.print_exception()
        else:
            console.print(f"\n[bold red]✗ Error:[/] {exc}")
            console.print("[dim]Run with --verbose for full traceback.[/]")
        return 2


if __name__ == "__main__":
    sys.exit(main())
