# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

datas = [("config/config.toml", "config")]
datas += collect_data_files("sentence_transformers")
datas += collect_data_files("transformers")

binaries = []
for package in ("faiss", "llama_cpp", "torch"):
    binaries += collect_dynamic_libs(package)

hiddenimports = []
for package in ("sentence_transformers", "transformers", "llama_cpp"):
    hiddenimports += collect_submodules(package)

a = Analysis(
    ["app/ui/__main__.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "tests", "notebook", "IPython", "jupyter"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="TechnicalDocumentIntelligence", debug=False,
    bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False,
)
coll = COLLECT(
    exe, a.binaries, a.datas, strip=False, upx=False,
    upx_exclude=[], name="TechnicalDocumentIntelligence",
)
