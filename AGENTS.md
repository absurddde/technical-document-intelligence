# Project Instructions

This project is an offline local document intelligence application.

Before making architectural changes, read REQUIREMENTS.md.

Core priorities:
1. Source accuracy
2. Traceability
3. Offline security
4. Retrieval quality
5. Maintainability

Development rules:
- Use Python.
- Keep the architecture modular.
- Use type hints.
- Add short docstrings to important functions.
- Do not introduce cloud APIs.
- Do not introduce telemetry or analytics.
- Do not automatically download models at runtime.
- All document processing must be local.
- All embeddings must be local.
- LLM inference must be local.
- Data must remain on the local machine.
- Prefer stable and widely used dependencies.
- Do not implement unrelated future features unless requested.
- Work incrementally.
- Run relevant tests after changes.
- Do not delete working functionality without a reason.

Important:
Do not attempt to implement the entire REQUIREMENTS.md in one task.
Implement only the requested development phase.