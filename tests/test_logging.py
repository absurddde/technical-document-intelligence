from pathlib import Path

from app.infrastructure.config import LoggingConfig
from app.infrastructure.logging import configure_logging


def test_rotating_local_logging_creates_utf8_log(tmp_path: Path) -> None:
    logger = configure_logging(
        tmp_path, LoggingConfig(level="INFO", max_bytes=512, backup_count=1)
    )
    logger.info("inventory event path=%s", "C:/archive/document.pdf")
    for handler in logger.handlers:
        handler.flush()

    content = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert "inventory event" in content
    assert "document content" not in content

