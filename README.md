# Technical Document Intelligence

A fully local Windows desktop RAG application for querying technical PDF and DOCX documents.

The project focuses on offline technical document analysis, source traceability, and privacy. Documents are processed locally and answers are generated from retrieved passages with file and page references.

## Features

- PDF and DOCX ingestion
- Offline OCR with Tesseract
- Multilingual embeddings with BGE-M3
- FAISS vector search
- SQLite FTS5 lexical search
- Hybrid semantic + lexical retrieval
- Reciprocal Rank Fusion (RRF)
- Local Qwen3-8B inference with llama.cpp
- Source and page-level citations
- Evidence passage inspection
- Insufficient-evidence detection
- Incremental indexing
- PySide6 Windows desktop interface
- Packaged Windows executable support
- Offline operation

## Tech Stack

- Python 3.11
- PySide6
- SQLite / FTS5
- FAISS
- BGE-M3
- Qwen3-8B GGUF
- llama-cpp-python
- Tesseract OCR
- PyInstaller
- pytest

## Local Models

Model files are not included in this repository because of their size.

Expected structure:

    models/
      embedding/
        bge-m3/
      llm/
        qwen3-8b/
          Qwen3-8B-Q4_K_M.gguf

## Source Documents

The PDF and DOCX documents used during development are not included in this repository.

Users are expected to provide their own local documents.

PDF and DOCX files are ignored by Git to prevent research papers, private documents, or licensed material from being committed accidentally.

## Installation

Clone the repository:

    git clone https://github.com/absurddde/technical-document-intelligence.git
    cd technical-document-intelligence

Create and activate a virtual environment:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

Tesseract OCR and the required local model files must also be installed separately.

## Run

    python -m app.ui

## Build for Windows

    .\scripts\build_windows.ps1

## Testing

    pytest -q

## Current Status

This project is a working MVP.

Core document ingestion, indexing, retrieval, local generation, citations, insufficient-evidence handling, and Windows packaging are functional.

## Known Limitations

- Domain-specific cross-language terminology can still cause retrieval failures, especially acronym pairs such as `İHA / İKA` and `UAV / UGV`.
- Qwen3-8B CPU inference can take several minutes depending on the retrieved context.
- Large local model files are not distributed through this repository.

## Goal

This project is designed as a local technical document intelligence system rather than a general-purpose chatbot. Answers are intended to remain grounded in and traceable to the user's own documents.
