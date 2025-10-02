import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "ct_clf_hack",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console_output: bool = True
) -> logging.Logger:
    logger = logging.getLogger(name)

    logger.handlers.clear()

    logger.setLevel(level)

    # Log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_default_logger() -> logging.Logger:
    log_dir = Path("logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ct_clf_hack_{timestamp}.log"

    return setup_logger(
        name="ct_clf_hack",
        level=logging.INFO,
        log_file=log_file,
        console_output=True
    )

default_logger = get_default_logger()
