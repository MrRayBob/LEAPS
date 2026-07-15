from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

from leaps import __version__
from leaps.ui.main_window import MainWindow
from leaps.ui.settings_dialog import FirstRunDialog
from leaps.ui.theme import APP_STYLESHEET, palette


def packaging_self_test() -> int:
    """Import critical modules and exercise packaged export and fitting."""
    import importlib.util

    import emcee
    import exotethys
    import h5py
    import numpy
    import photutils.geometry.core as photutils_geometry
    import pyvo.samp as pyvo_samp
    import quantities
    import requests
    import scipy.ndimage
    import scipy.optimize
    import yaml
    from astropy.io import fits
    from astropy.wcs import WCS
    from astroquery.gaia import Gaia
    from matplotlib.backends import backend_agg, backend_pdf
    from matplotlib.figure import Figure
    from photutils.aperture import CircularAperture, aperture_photometry
    from PIL import Image

    import hops.pylightcurve41 as plc
    from hops.hops_tools import image_analysis

    exoclock_spec = importlib.util.find_spec("exoclock")

    required = (
        emcee,
        h5py,
        numpy,
        photutils_geometry,
        quantities,
        requests,
        scipy.ndimage,
        scipy.optimize,
        yaml,
        fits,
        WCS,
        Gaia,
        image_analysis,
        backend_agg,
        backend_pdf,
        CircularAperture,
        aperture_photometry,
        Image,
        plc,
        exotethys,
        pyvo_samp,
        exoclock_spec,
    )
    if any(item is None for item in required):
        return 1
    with tempfile.TemporaryDirectory(prefix="leaps-packaging-test-") as directory:
        figure = Figure(figsize=(2, 2))
        axis = figure.add_subplot(111)
        axis.plot((0, 1), (0, 1))
        png_path = Path(directory) / "figure.png"
        pdf_path = Path(directory) / "figure.pdf"
        figure.savefig(png_path)
        figure.savefig(pdf_path)
        if not png_path.is_file() or not pdf_path.is_file():
            return 1
        if _packaging_fitting_self_test(Path(directory)) != 0:
            return 1
    print("LEAPS packaged runtime self-test passed", flush=True)
    return 0


def _packaging_fitting_self_test(directory: Path) -> int:
    """Run the LEAPS/HOPS preview path inside the packaged interpreter."""
    import numpy as np

    import hops.pylightcurve41 as plc
    from hops.pylightcurve41.models.exoplanet import Planet
    from leaps.catalog import PlanetParameters
    from leaps.project import ProjectWorkspace
    from leaps.science import FittingService

    original_exotethys = Planet.exotethys
    original_fp_over_fs = Planet.fp_over_fs
    original_all_filters = plc.all_filters
    plc.all_filters = lambda: ["COUSINS_R"]

    def fixed_flux_ratio(
        self: Planet,
        filter_name: str,
        wlrange: list[float] | None = None,
    ) -> float:
        del self, filter_name, wlrange
        return 0.0

    Planet.fp_over_fs = fixed_flux_ratio
    def fixed_limb_darkening(
        self: Planet,
        filter_name: str,
        wlrange: list[float] | None = None,
        stellar_model: str = "Phoenix_2018",
    ) -> np.ndarray:
        del self, filter_name, wlrange, stellar_model
        return np.asarray((0.60, -0.10, 0.05, -0.02), dtype=float)

    Planet.exotethys = fixed_limb_darkening

    try:
        project = ProjectWorkspace.create(directory / "fit-project", "Package fitting test")
        light_curve_dir = project.outputs_dir / "light_curve"
        light_curve_dir.mkdir(parents=True, exist_ok=True)
        times = np.linspace(2461172.70, 2461172.92, 96)
        transit = 0.021 * np.exp(-((times - 2461172.8555) / 0.035) ** 4)
        flux = 1.0 - transit + 0.0008 * np.sin(np.linspace(0, 6 * np.pi, times.size))
        np.savetxt(
            light_curve_dir / "light_curve_gauss.txt",
            np.column_stack((times, flux, np.full(times.size, 0.0015))),
        )
        parameters = PlanetParameters(
            name="LEAPS package test",
            ra="18:57:35.94",
            dec="-49:08:18.65",
            period=3.18,
            mid_time=2461172.8555,
            rp_over_rs=0.1456,
            sma_over_rs=10.0,
            inclination=86.17,
            eccentricity=0.0,
            periastron=0.0,
            metallicity=0.0,
            temperature=5500.0,
            logg=4.5,
            source="Package self-test",
            is_manual=True,
        )
        result = FittingService().run(
            project,
            parameters,
            full=False,
            exposure_time=120.0,
            filter_name="COUSINS_R",
            latitude=-30.33667,
            longitude=-70.79992,
            light_curve="gaussian",
            detrending="quadratic",
        )
        if not result.preview_path.is_file() or result.residual_std is None:
            raise RuntimeError("The packaged HOPS preview did not produce a complete result")
    except BaseException:
        diagnostic = os.getenv("LEAPS_PACKAGING_DIAGNOSTIC_PATH")
        if diagnostic:
            Path(diagnostic).write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        return 1
    finally:
        Planet.exotethys = original_exotethys
        Planet.fp_over_fs = original_fp_over_fs
        plc.all_filters = original_all_filters
    return 0


def create_application(argv: list[str] | None = None) -> QApplication:
    QApplication.setOrganizationName("LEAPS")
    QApplication.setOrganizationDomain("leaps-astronomy.org")
    QApplication.setApplicationName("LEAPS")
    QApplication.setApplicationVersion(__version__)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, False)
    app = QApplication.instance() or QApplication(argv or sys.argv)
    if sys.platform == "darwin":
        app.setFont(QFont(".AppleSystemUIFont"))
    elif sys.platform.startswith("win"):
        app.setFont(QFont("Segoe UI"))
    app.setPalette(palette())
    app.setStyleSheet(APP_STYLESHEET)
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / "assets" / "leaps-app-icon.png")))
    return app


def main() -> int:
    if "--packaging-self-test" in sys.argv:
        return packaging_self_test()
    app = create_application()
    demo = os.getenv("LEAPS_DEMO") == "1"
    window = MainWindow(demo=demo)
    window.show()
    if not demo and os.getenv("LEAPS_SKIP_ONBOARDING") != "1":
        settings = QSettings()
        if not settings.value("setup/complete", False, type=bool):
            setup = FirstRunDialog(window)
            if setup.exec():
                settings.setValue("setup/complete", True)
                if setup.offline_choice.currentIndex() == 1:
                    QTimer.singleShot(0, window.open_settings)
    screenshot = os.getenv("LEAPS_SCREENSHOT_PATH")
    if screenshot:

        def capture() -> None:
            path = Path(screenshot).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(path))
            app.quit()

        QTimer.singleShot(1000, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
