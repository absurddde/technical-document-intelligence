# Technical Document Intelligence

Fully local Windows desktop document retrieval and grounded synthesis for PDF and DOCX archives.

## Windows distribution

The supported packaging target is Windows 10/11 64-bit. A 64-bit CPU, 16 GB RAM, and an SSD are recommended. CPU-only operation is supported; Qwen3 CPU responses can take several minutes.

Run `TechnicalDocumentIntelligence.exe` from the distribution folder. VS Code, the source repository, and a developer Python installation are not intended to be required by the packaged application.

Models are deliberately external and are never downloaded at runtime:

```text
TechnicalDocumentIntelligence/
  TechnicalDocumentIntelligence.exe
  _internal/
  config/config.toml
  models/embedding/bge-m3/
  models/llm/qwen3-8b/Qwen3-8B-Q4_K_M.gguf
```

Copy the complete BGE-M3 directory and Qwen3 GGUF to those locations before indexing or asking questions. Do not move files out of `_internal`.

Tesseract OCR is required only for scanned PDFs. This release supports either an existing Tesseract installation (PATH or the standard Windows install directory) or a separately supplied `Tesseract-OCR` folder beside the EXE. Turkish (`tur`) and English (`eng`) language data must be installed. Tesseract binaries are not included by this project.

## Use

Open the application, add PDF/DOCX files, and run indexing. Enter a technical question after indexing to retrieve local evidence and generate a cited Turkish answer. Original documents remain in their selected locations and are not modified.

All runtime processing is offline. The application has no cloud API, login, telemetry, analytics, remote database, or automatic model download. Mutable state is stored under `%LOCALAPPDATA%\TechnicalDocumentIntelligence\`:

- `database/app.db`: document metadata and lexical/embedding state
- `indexes/`: FAISS artifacts
- `cache/`: parsed-document cache
- `logs/app.log`: bounded rotating operational logs

If an original document is moved or deleted, add or rescan the current file location. A missing FAISS artifact can be recreated by indexing again. A changed embedding model fingerprint causes embeddings/index state to be rebuilt rather than silently reused. Interrupted indexing retains committed SQLite state and can be safely rerun. Do not delete the LOCALAPPDATA folder unless intentionally resetting all application state; back it up before manual recovery from database corruption.

Known limitations: models and Tesseract must be supplied separately; DOCX page numbers are not always available; CPU generation is slow; clean-machine operation must be validated on the target Windows configuration.

## Development and build

Development mode remains:

```powershell
D:\proje_Staj\.venv\Scripts\python.exe -m app.ui
```

After installing PyInstaller in the project venv, build an onedir distribution with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1 -Clean
```

Output is written to `dist\TechnicalDocumentIntelligence\`; large models, user data, pilot documents/reports, and development caches are excluded.
