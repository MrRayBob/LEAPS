from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from leaps.exports import TransitExporter
from leaps.models import StageID
from leaps.project import ProjectWorkspace


def test_exoclock_and_etd_exports_are_created_from_last_success(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path, "Export test")
    project.manifest.target_name = "WTS-2 b"
    project.manifest.target_ra = "19:34:55.87"
    project.manifest.target_dec = "+36:48:55.79"
    output = project.outputs_dir / StageID.LIGHT_CURVE.value
    output.mkdir()
    curve = np.array([[2461000.1, 1.0, 0.002], [2461000.2, 0.97, 0.0025]])
    np.savetxt(output / "light_curve_aperture.txt", curve)
    exporter = TransitExporter(project)
    exoclock = exporter.export_exoclock(tmp_path / "exoclock.txt")
    etd = exporter.export_etd(tmp_path / "etd.txt")
    assert exoclock.exists() and etd.exists()
    assert np.loadtxt(exoclock).shape == (2, 3)
    etd_data = np.loadtxt(etd)
    assert etd_data[1, 1] > 0
    metadata = json.loads((tmp_path / "exoclock.txt.json").read_text())
    assert metadata["target"] == "WTS-2 b"
    assert metadata["format"] == "ExoClock"
