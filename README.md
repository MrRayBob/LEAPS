# LEAPS

LEAPS is a friendly desktop workflow for reducing and analysing exoplanet-transit FITS sequences. It preserves the established HOPS scientific routines while replacing the complete Tkinter interface with a single, stable PySide6 application.

The first release focuses on reliability and approachability:

- one resizable window for Data & Target, Reduction, Inspection, Alignment, Photometry, Light Curve, Fitting, and Secondary Eclipse;
- coordinates as the canonical target identity, with target name optional;
- HOPS-style target-name lookup through ExoClock/SIMBAD, with offline NASA and reusable cache fallbacks;
- automatic FITS frame classification with explicit calibration waivers;
- background processing, safe cancellation, checkpoints, and continuous project autosave;
- real reduced-FITS viewing with working pan, zoom, invert, reset, and target/comparison overlays;
- optional plate solving that validates an existing FITS WCS first, then uses bounded Gaia attempts with a visible timeline and manual target placement fallback;
- the original HOPS star detection, Gaussian fitting, geometric-centering option, variable aperture, sky-annulus, and differential-photometry calculations behind the new interface;
- a required Light Curve review where anomalous comparison stars can be excluded before HOPS-compatible fitting and export;
- target-coordinate-driven planet defaults, automatic HOPS passband/exposure detection, a rendered Preview Fit gate, and transactional full-fit outputs;
- a fixed-phase secondary-eclipse (occultation) workflow that reuses the completed fit, reports red-noise-aware uncertainty and nearby control-phase checks, and clearly distinguishes candidate, marginal, and inconclusive results;
- typed, recoverable failures and one-click redacted diagnostic ZIP export;
- offline-data management with size estimates, disk checks, resumable downloads, and project-region Gaia packages;
- familiar HOPS-compatible outputs plus ExoClock and ETD export surfaces;
- a signed Apple-Silicon DMG and signed Windows 10/11 x64 installer build pipeline.

## Run from source

LEAPS supports Python 3.11–3.14.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m leaps
```

On Windows, activate the environment with `.venv\Scripts\activate` before running the same install and launch commands.

Raw FITS files are opened read-only. LEAPS creates a visible, portable `LEAPS/` folder inside the observing run containing `project.json`, structured logs, caches, checkpoints, and generated outputs. Moving the run folder between macOS and Windows preserves relative project references. Existing hidden `.leaps/` projects are validated and migrated automatically when opened. Data & Target can reveal this folder or safely reset only LEAPS-generated data after exact-name confirmation.

To resume an existing run, use **Data & Target → Open project** and choose the observing-run folder containing `LEAPS/project.json` (or the `LEAPS` folder itself). LEAPS restores the saved workflow state without rescanning or changing raw FITS files.

## TESS light-curve import

Use **Data & Target → Import TESS light curves** to select one or more downloaded TESS SPOC `*_lc.fits` products for a single TIC target. LEAPS reads their calibrated `PDCSAP_FLUX` values and mission quality flags without changing the source files, creates a portable `TESS-TIC-<id>/LEAPS/` project beside the selected data, and opens **Fitting**.

The primary-transit fit is catalog-guided: LEAPS refines the known ephemeris with Box Least Squares, phase-folds the many TESS epochs into one HOPS-compatible transit fit, and then enables **Secondary Eclipse** after a completed full fit. TESS target-pixel files and full-frame images are deliberately not accepted here; they require a separate photometric-extraction workflow before fitting.

## Secondary eclipse analysis

Run a full primary-transit fit first, then open **Secondary Eclipse**. LEAPS carries forward the saved ephemeris and approved aperture/PSF light curve, suggests a duration from the transit geometry, and evaluates only the expected occultation phase (normally 0.50 for a circular orbit). It also fits nearby control phases and scales the depth uncertainty for time-correlated noise.

Each run writes a plot/PDF, the local phase-folded CSV, and a JSON summary to `LEAPS/outputs/secondary_eclipse/`. A candidate signal is not a confirmation: it requires independent eclipse coverage and should not be interpreted as an albedo measurement without an emission/reflection model. If the observation does not cover phase 0.5, LEAPS records an explicit inconclusive result rather than fitting a false depth.

### Optional ML recovery validation

For an imported TESS project with at least four readable sectors, a saved eclipse result unlocks **Secondary Eclipse → Optional ML recovery check**. This is intentionally a validation experiment—not an automatic discovery button. LEAPS removes the fitted real eclipse from each sector's local residuals, injects known fake eclipses plus off-phase decoy dips, trains a small random forest on one group of sectors, calibrates its threshold on another, and evaluates it only on held-out sectors. Alongside the usual fixed-phase depth and red-noise features, the classifier records whether a positive, mutually consistent depth recurs across independent sectors; this is a physical repeatability check, not a substitute for the eclipse fit.

The exported `LEAPS/outputs/secondary_eclipse_ml/` folder contains a recovery curve, feature-importance plot, row-level trial table, and reproducibility summary. The ML score is always paired with LEAPS' positive-depth and nearby-control safety guard; it never changes the normal candidate/marginal/inconclusive outcome. Install the optional dependency when running from source:

```bash
python -m pip install -e '.[ml]'
```

### Cross-target reliability study

After running the current ML recovery check for at least three TESS targets, open one target's **Secondary Eclipse** page and choose **Run cross-target study**. LEAPS includes that target automatically; select the other targets' `LEAPS/outputs/secondary_eclipse_ml/ml-trials.csv` files. For each planet, it trains and sets the score threshold only on the other planets, then tests the held-out target. This leave-one-planet-out result is exported to `LEAPS/outputs/secondary_eclipse_cross_target_ml/`.

It is deliberately harder than a held-out-sector result. A lower recovery floor only counts as an improvement if the held-out false-alarm rate does not increase. Use it to assess whether a frozen reliability feature set transfers across targets, not to make an ML-only eclipse claim.

## Test

```bash
python -m pip install -e '.[dev]'
QT_QPA_PLATFORM=offscreen pytest
```

## Package

Qt's supported `pyside6-deploy` workflow is configured in `pysidedeploy.spec`.

```bash
bash scripts/build_macos.sh
```

On Windows PowerShell:

```powershell
./scripts/build_windows.ps1
```

Unsigned local artifacts can be built without credentials. Signed release artifacts require the Apple Developer ID/notarization and Windows code-signing secrets documented by the inputs in `.github/workflows/release.yml`. The workflow creates draft GitHub releases so a human can validate installers before publication.

## Design and validation

The approved workspace reference is saved at `docs/design/leaps-plate-solve-reference.png`. The production workspace now replaces the reference's illustrative WTS-2 b field with the project's actual reduced FITS frame and target identity. Rendered implementation comparisons and the design acceptance record are stored in `docs/design/` and `design-qa.md`.

The temporary LEAPS wordmark and mark are centralized under `leaps/assets/`; they can be replaced when the final team logo is ready without restructuring the interface.

## Scope

LEAPS v1 is not a broader astronomy suite. Linux installers, Intel Mac builds, blind phase searches, telemetry, and a user-facing CLI are intentionally out of scope. The observing planner remains available under Tools.

This fork retains upstream HOPS scientific code and licensing notices. See [LICENSE](LICENSE).
