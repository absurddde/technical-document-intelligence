"""Development entry point: python -m app.ui."""

from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from app.infrastructure.config import apply_runtime_paths, load_config
from app.infrastructure.health import (
    format_health_problems, run_startup_health, with_discovered_tesseract,
)
from app.infrastructure.logging import configure_logging
from app.infrastructure.paths import discover_runtime_paths, enforce_offline_environment
from app.ui.backend import LocalBackendFacade
from app.ui.main_window import MainWindow


def main() -> int:
    """Start the fully local Phase 6 desktop application."""

    runtime = discover_runtime_paths()
    enforce_offline_environment()
    smoke_test = "--smoke-test" in sys.argv
    if smoke_test:
        sys.argv.remove("--smoke-test")
    application = QApplication(sys.argv)
    application.setApplicationName("Technical Document Intelligence")
    application.setApplicationDisplayName("Technical Document Intelligence")
    application.setOrganizationName("TechnicalDocumentIntelligence")
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
        config = apply_runtime_paths(load_config(runtime.config_file), runtime)
        config = with_discovered_tesseract(config, runtime)
        configure_logging(config.paths.logs, config.logging)
        health = run_startup_health(config, runtime)
        window = MainWindow(LocalBackendFacade(config))
    except Exception as error:
        QMessageBox.critical(None, "Başlatma hatası", str(error))
        return 1
    window.show()
    problems = format_health_problems(health)
    if problems and not smoke_test:
        QMessageBox.warning(
            window, "Startup health check",
            "Some local components are unavailable. Affected operations may fail:\n\n"
            + problems,
        )
    if smoke_test:
        QTimer.singleShot(1000, application.quit)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
