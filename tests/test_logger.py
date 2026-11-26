# ===== LOGGER TEST START =====
# Checks the shared logger in src/logger.py. Run with: pytest -q tests/test_logger.py
import logging

from src.logger import get_logger


def test_get_logger_returns_logger_with_handlers():
    logger = get_logger("test_logger_handlers")
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) >= 1


def test_get_logger_does_not_add_handlers_twice():
    first = get_logger("test_logger_twice")
    count = len(first.handlers)
    second = get_logger("test_logger_twice")
    assert first is second
    assert len(second.handlers) == count
# ===== LOGGER TEST END =====
