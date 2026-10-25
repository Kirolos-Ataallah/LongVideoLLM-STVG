from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler

_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, show_path=False, log_time_format="[%X]")
    ]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT))
        handlers.append(fh)
    logging.basicConfig(level=level, format="%(message)s", handlers=handlers, force=True)
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub", "qwen_vl_utils"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
