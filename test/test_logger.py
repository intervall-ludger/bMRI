from __future__ import annotations

import logging

from bmri.logger import get_logger, setup_logging


def test_setup_logging_with_file(tmp_path) -> None:
    log_file = tmp_path / "bmri.log"
    logger = setup_logging(level="INFO", log_file=log_file)
    logger.info("hello logger")

    assert log_file.exists()
    contents = log_file.read_text()
    assert "hello logger" in contents

    module_logger = get_logger("bmri.test")
    assert isinstance(module_logger, logging.Logger)
