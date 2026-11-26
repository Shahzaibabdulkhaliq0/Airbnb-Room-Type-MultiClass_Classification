# ===== LOGGER START: shared logger for the whole project =====
# How to use in any file:
#   from src.logger import get_logger
#   logger = get_logger(__name__)
#   logger.info("message")
# Output goes to the console AND to logs/app.log (logs/ is in .gitignore).

import logging
from pathlib import Path

# The project folder is one folder above this file (src/ -> project root)
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name):
    logger = logging.getLogger(name)

    # Already set up? Return it. This stops the same log line from printing twice.
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler: shows logs in the terminal (and in GitHub Actions)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler: saves logs to logs/app.log. If the folder cannot be made, only the console is used.
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("Could not create the log file; logging to console only.")

    return logger
# ===== LOGGER END =====
