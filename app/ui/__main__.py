"""Development entry point: python -m app.ui."""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from app.infrastructure.config import load_config
from app.infrastructure.logging import configure_logging
from app.ui.backend import LocalBackendFacade
from app.ui.main_window import MainWindow


def main() -> int:
    """Start the fully local Phase 6 desktop application."""

    root = Path(__file__).resolve().parents[2]
    application = QApplication(sys.argv)
    application.setApplicationName("Yerel Teknik Doküman Zekâsı")
    application.setStyleSheet("""
        QWidget { font-family: "Segoe UI"; font-size: 10pt; }
        QLabel#title { font-size: 20pt; font-weight: 600; }
        QLabel#subtitle, QLabel#hint { color: #5f6b76; }
        QPushButton { padding: 7px 12px; }
        QPushButton#primary { background: #1769aa; color: white; border: 0; border-radius: 4px; }
        QPushButton#primary:disabled { background: #9aa7b2; }
        QPlainTextEdit, QTextBrowser, QTableWidget, QListWidget, QTreeWidget {
            border: 1px solid #cbd3da; border-radius: 4px; background: white;
        }
    """)
    try:
        config = load_config(root / "config/config.toml")
        configure_logging(config.paths.logs, config.logging)
        window = MainWindow(LocalBackendFacade(config))
    except Exception as error:
        QMessageBox.critical(None, "Başlatma hatası", str(error))
        return 1
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
