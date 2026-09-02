from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os

from app.infrastructure.config import apply_runtime_paths, load_config
from app.infrastructure.health import discover_tesseract, run_startup_health
from app.infrastructure.paths import discover_runtime_paths, enforce_offline_environment


def test_development_paths_remain_project_relative(tmp_path: Path) -> None:
    module = tmp_path / "app" / "infrastructure" / "paths.py"
    runtime = discover_runtime_paths(frozen=False, module_file=module)
    assert runtime.application_dir == tmp_path
    assert runtime.database == tmp_path / "data" / "database" / "app.db"
    assert runtime.models == tmp_path / "models"


def test_packaged_paths_separate_resources_models_and_user_data(tmp_path: Path) -> None:
    app_dir = tmp_path / "distribution"
    resources = app_dir / "_internal"
    config = resources / "config" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text("", encoding="utf-8")
    local = tmp_path / "local"
    runtime = discover_runtime_paths(
        frozen=True, executable=app_dir / "TechnicalDocumentIntelligence.exe",
        bundle_dir=resources, local_app_data=local,
    )
    assert runtime.config_file == config
    assert runtime.models == app_dir / "models"
    assert runtime.database == local / "TechnicalDocumentIntelligence" / "database" / "app.db"


def test_packaged_configuration_uses_external_models_and_localappdata(tmp_path: Path) -> None:
    config = load_config(Path(__file__).parents[1] / "config" / "config.toml")
    runtime = discover_runtime_paths(
        frozen=True, executable=tmp_path / "app" / "TechnicalDocumentIntelligence.exe",
        bundle_dir=tmp_path / "app" / "_internal", local_app_data=tmp_path / "user",
    )
    packaged = apply_runtime_paths(config, runtime)
    assert packaged.embedding.model_path == runtime.models / "embedding" / "bge-m3"
    assert packaged.llm.model_path.suffix == ".gguf"
    assert packaged.paths.logs == runtime.logs


def test_application_local_tesseract_precedes_path(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "Tesseract-OCR" / "tesseract.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"test")
    monkeypatch.setattr("shutil.which", lambda _: "C:/other/tesseract.exe")
    assert discover_tesseract(None, tmp_path) == executable


def test_health_reports_missing_models_without_loading_them(tmp_path: Path, monkeypatch) -> None:
    source_config = load_config(Path(__file__).parents[1] / "config" / "config.toml")
    config = replace(
        source_config,
        paths=replace(
            source_config.paths, database=tmp_path / "data" / "database" / "app.db",
            indexes=tmp_path / "data" / "indexes", cache=tmp_path / "data" / "cache",
            logs=tmp_path / "data" / "logs",
        ),
        embedding=replace(source_config.embedding, model_path=tmp_path / "missing-embedding"),
        llm=replace(source_config.llm, model_path=tmp_path / "missing.gguf"),
        ocr=replace(source_config.ocr, enabled=False),
    )
    runtime = discover_runtime_paths(frozen=False, module_file=tmp_path / "app/x/paths.py")
    monkeypatch.setattr("app.infrastructure.health.discover_tesseract", lambda *_: None)
    monkeypatch.setattr("app.infrastructure.health.importlib.import_module", lambda _: object())
    checks = run_startup_health(config, runtime)
    states = {item.name: item.ok for item in checks}
    assert states["Data directory"] and states["SQLite"]
    assert not states["BGE-M3"] and not states["Qwen3 GGUF"]


def test_offline_environment_is_forced(monkeypatch) -> None:
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
        monkeypatch.delenv(name, raising=False)
    enforce_offline_environment()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
