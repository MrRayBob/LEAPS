from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .models import LEAPSError, StageID
from .project import ProjectWorkspace


class TransitExporter:
    def __init__(self, project: ProjectWorkspace) -> None:
        self.project = project

    def _curve(self) -> np.ndarray:
        path = self.project.outputs_dir / StageID.LIGHT_CURVE.value / "light_curve_aperture.txt"
        if not path.exists():
            raise LEAPSError(
                "LIGHT_CURVE_REQUIRED",
                "No successful light curve is available",
                "Review and approve the comparison-star ensemble before creating an export.",
                ["Open Light Curve", "Confirm the comparison stars"],
                stage=StageID.LIGHT_CURVE,
            )
        curve = np.atleast_2d(np.loadtxt(path))
        if curve.shape[1] < 3:
            raise LEAPSError(
                "LIGHT_CURVE_FORMAT",
                "The light curve has an unexpected format",
                "Export diagnostics so the input columns can be reviewed.",
                ["Export diagnostics"],
            )
        return curve[:, :3]

    def export_exoclock(self, destination: str | Path) -> Path:
        destination = Path(destination)
        curve = self._curve()
        np.savetxt(
            destination, curve, fmt=["%.8f", "%.8f", "%.8f"], header="JD_UTC NORMALISED_FLUX FLUX_UNCERTAINTY"
        )
        self._metadata(destination.with_suffix(destination.suffix + ".json"), "ExoClock")
        return destination

    def export_etd(self, destination: str | Path) -> Path:
        destination = Path(destination)
        curve = self._curve()
        safe_flux = np.clip(curve[:, 1], 1e-12, None)
        magnitude = -2.5 * np.log10(safe_flux)
        magnitude_error = (2.5 / math.log(10)) * np.abs(curve[:, 2] / safe_flux)
        output = np.column_stack((curve[:, 0], magnitude, magnitude_error))
        np.savetxt(
            destination,
            output,
            fmt=["%.8f", "%.8f", "%.8f"],
            header="JD_UTC DIFFERENTIAL_MAGNITUDE MAGNITUDE_UNCERTAINTY",
        )
        self._metadata(destination.with_suffix(destination.suffix + ".json"), "ETD")
        return destination

    def _metadata(self, destination: Path, format_name: str) -> None:
        payload = {
            "format": format_name,
            "target": self.project.manifest.target_name,
            "ra_icrs": self.project.manifest.target_ra,
            "dec_icrs": self.project.manifest.target_dec,
            "project_id": self.project.manifest.project_id,
            "warnings": self.project.manifest.warnings,
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
