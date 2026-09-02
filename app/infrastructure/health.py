"""Lightweight startup checks that never load model weights."""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
import os
from pathlib import Path
import shutil
import sqlite3

from app.infrastructure.config import AppConfig
from app.infrastructure.paths import RuntimePaths, ensure_local_directories


@dataclass(frozen=True, slots=True)
class HealthCheck:
    """One startup dependency result."""

    name: str
    ok: bool
    message: str
    critical: bool = False


def discover_tesseract(configured: str | None, application_dir: Path) -> Path | None:
    """Find configured, application-local, PATH, or standard Windows Tesseract."""

    candidates: list[Path] = []
    if configured:
        value = Path(configured).expanduser()
        candidates.append(value if value.is_absolute() else application_dir / value)
    candidates.extend([
        application_dir / "Tesseract-OCR" / "tesseract.exe",
        application_dir / "tesseract" / "tesseract.exe",
    ])
    located = shutil.which("tesseract")
    if located:
        candidates.append(Path(located))
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(variable)
        if root:
            candidates.append(Path(root) / "Tesseract-OCR" / "tesseract.exe")
    return next((item.resolve(strict=False) for item in candidates if item.is_file()), None)


def with_discovered_tesseract(config: AppConfig, runtime: RuntimePaths) -> AppConfig:
    """Return configuration using the discovered local OCR executable."""

    command = discover_tesseract(config.ocr.tesseract_command, runtime.application_dir)
    return replace(config, ocr=replace(
        config.ocr, tesseract_command=str(command) if command else None,
    ))


def run_startup_health(config: AppConfig, runtime: RuntimePaths) -> tuple[HealthCheck, ...]:
    """Check local runtime prerequisites without instantiating ML models."""

    results: list[HealthCheck] = []
    try:
        ensure_local_directories([
            config.paths.database.parent, config.paths.indexes,
            config.paths.cache, config.paths.logs,
        ])
        probe = runtime.user_data_dir / ".write-test"
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
        results.append(HealthCheck("Data directory", True, str(runtime.user_data_dir), True))
    except OSError as error:
        results.append(HealthCheck("Data directory", False, str(error), True))
    try:
        connection = sqlite3.connect(config.paths.database)
        connection.execute("SELECT 1").fetchone()
        connection.close()
        results.append(HealthCheck("SQLite", True, "available", True))
    except sqlite3.Error as error:
        results.append(HealthCheck("SQLite", False, str(error), True))
    results.append(HealthCheck(
        "BGE-M3", config.embedding.model_path.is_dir(), str(config.embedding.model_path),
    ))
    results.append(HealthCheck(
        "Qwen3 GGUF", config.llm.model_path.is_file(), str(config.llm.model_path),
    ))
    tesseract = discover_tesseract(config.ocr.tesseract_command, runtime.application_dir)
    if not tesseract:
        results.append(HealthCheck("Tesseract", False, "executable not found"))
    else:
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = str(tesseract)
            languages = set(pytesseract.get_languages(config=""))
            missing = sorted(set(config.ocr.languages) - languages)
            message = str(tesseract)
            if missing:
                message += "; missing languages: " + ", ".join(missing)
            results.append(HealthCheck("Tesseract OCR languages", not missing, message))
        except Exception as error:
            results.append(HealthCheck("Tesseract", False, str(error)))
    for label, module, critical in (
        ("FAISS", "faiss", False), ("llama.cpp", "llama_cpp", False),
        ("Qt runtime", "PySide6.QtCore", True),
    ):
        try:
            importlib.import_module(module)
            results.append(HealthCheck(label, True, "available", critical))
        except Exception as error:
            results.append(HealthCheck(label, False, str(error), critical))
    return tuple(results)


def format_health_problems(checks: tuple[HealthCheck, ...]) -> str:
    """Format only failed checks for a concise UI warning."""

    return "\n".join(f"- {item.name}: {item.message}" for item in checks if not item.ok)
