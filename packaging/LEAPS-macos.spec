# -*- mode: python ; coding: utf-8 -*-
"""Dependency-complete PyInstaller bundle for Apple Silicon releases."""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files


ROOT = Path(SPECPATH).parent
VERSION = os.environ.get("LEAPS_VERSION", "2.1.1").removeprefix("v")
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
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "leaps" / "assets" / "leaps-app-icon.png"),
)

bundle_files = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="LEAPS",
)

application = BUNDLE(
    bundle_files,
    name="LEAPS.app",
    icon=str(ROOT / "leaps" / "assets" / "leaps-app-icon.png"),
    bundle_identifier="org.leaps.exoplanet",
    info_plist={
        "CFBundleDisplayName": "LEAPS",
        "CFBundleName": "LEAPS",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "NSDocumentsFolderUsageDescription": "LEAPS needs access to your observing-run folder to read FITS images and save project results beside them.",
        "NSDesktopFolderUsageDescription": "LEAPS needs access when an observing run is stored on your Desktop.",
        "NSDownloadsFolderUsageDescription": "LEAPS needs access when an observing run is stored in Downloads.",
        "NSNetworkVolumesUsageDescription": "LEAPS needs access when an observing run is stored on a network volume.",
        "NSRemovableVolumesUsageDescription": "LEAPS needs access when an observing run is stored on a removable drive.",
    },
)
