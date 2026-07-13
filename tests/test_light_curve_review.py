from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from leaps.models import ProjectManifest, StageID, StageStatus
from leaps.project import ProjectWorkspace
from leaps.science import LightCurveReviewService


def _project_with_measurements(tmp_path: Path) -> ProjectWorkspace:
    project = ProjectWorkspace.create(tmp_path, "Comparison review")
    photometry = project.outputs_dir / StageID.PHOTOMETRY.value
    photometry.mkdir()
    target = [100.0, 98.0, 101.0, 99.0]
    c1 = [50.0, 50.0, 50.0, 50.0]
    c2 = [50.0, 100.0, 50.0, 50.0]
    c3 = [50.0, 50.0, 50.0, 50.0]
    rows = []
    for index, values in enumerate(zip(target, c1, c2, c3, strict=True)):
        rows.append(
            {
                "file": f"light_{index:03d}.fits",
                "jd": 2461000.0 + index / 1440,
                "measurements": [
                    {
                        "aperture_flux": value,
                        "aperture_error": 1.0,
                        "gaussian_flux": value * 0.98,
                        "gaussian_error": 1.1,
                    }
                    for value in values
                ],
            }
        )
    (photometry / "measurements.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    project.manifest.stages[StageID.PHOTOMETRY.value].status = StageStatus.COMPLETE
    project.manifest.stages[StageID.LIGHT_CURVE.value].status = StageStatus.READY
    project.save()
    return project


def test_review_defaults_all_comparisons_on_and_plots_target_first(tmp_path: Path) -> None:
    project = _project_with_measurements(tmp_path)

    result = LightCurveReviewService().load(project)

    assert result.active_comparisons == [True, True, True]
    assert [curve.label for curve in result.curves] == ["Target", "C1", "C2", "C3"]
    assert all(curve.active for curve in result.curves)
    assert result.preview_path.exists()
    assert result.frame_count == 4


def test_excluding_anomalous_comparison_rebuilds_approved_curve(tmp_path: Path) -> None:
    project = _project_with_measurements(tmp_path)
    service = LightCurveReviewService()
    all_active = service.load(project)

    output = service.commit(project, [True, False, True])

    approved = np.loadtxt(output)
    expected = np.array([100.0, 98.0, 101.0, 99.0]) / 100.0
    expected /= np.median(expected)
    assert np.allclose(approved[:, 1], expected)
    assert not np.allclose(approved[:, 1], all_active.curves[0].aperture[:, 1])
    review = json.loads((output.parent / "review.json").read_text(encoding="utf-8"))
    assert review["active_comparisons"] == [True, False, True]
    assert review["active_labels"] == ["C1", "C3"]
    assert (output.parent / "light-curves.png").exists()
    assert (project.outputs_dir / StageID.PHOTOMETRY.value / "measurements.json").exists()


def test_manifest_v1_migration_inserts_required_light_curve_review() -> None:
    payload = ProjectManifest().to_dict()
    payload["schema_version"] = 1
    payload["stages"].pop(StageID.LIGHT_CURVE.value)
    payload["stages"][StageID.PHOTOMETRY.value]["status"] = "complete"
    payload["stages"][StageID.PHOTOMETRY.value]["summary"] = "Complete"

    migrated = ProjectManifest.from_dict(payload)

    assert migrated.schema_version == 2
    assert migrated.stages[StageID.LIGHT_CURVE.value].status == StageStatus.READY
    assert migrated.stages[StageID.LIGHT_CURVE.value].summary == "Review comparison stars"
