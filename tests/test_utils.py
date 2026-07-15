import logging

from aqua_align.utils import setup_logger


def test_setup_logger_is_idempotent() -> None:
    first = setup_logger("aqua_align.test", level="DEBUG")
    second = setup_logger("aqua_align.test", level="INFO")
    assert first is second
    assert second.level == logging.INFO
    assert len(second.handlers) == 1
