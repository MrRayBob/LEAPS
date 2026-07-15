# -*- mode: python ; coding: utf-8 -*-
"""Dependency-complete PyInstaller bundle for the Windows release runner."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files


ROOT = Path(SPECPATH).parent
datas = [
    (str(ROOT / "leaps" / "assets"), "leaps/assets"),
    (str(ROOT / "leaps" / "assets"), "assets"),
]
binaries = []
hiddenimports = [
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_pdf",
]

for package in ("hops", "exotethys", "photutils"):
    package_datas, package_binaries, package_imports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_imports

datas += collect_data_files("exoclock")
datas += collect_data_files("astroquery")
datas += collect_data_files("pyvo")

analysis = Analysis(
    [str(ROOT / "leaps" / "app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="LEAPS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=str(ROOT / "leaps" / "assets" / "leaps-app-icon.png"),
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="LEAPS",
)
