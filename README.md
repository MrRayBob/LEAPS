# LEAPS

LEAPS is a friendly desktop workflow for reducing and analysing exoplanet-transit FITS sequences. It preserves the established HOPS scientific routines while replacing the complete Tkinter interface with a single, stable PySide6 application.

The first release focuses on reliability and approachability:

- one resizable window for Data & Target, Reduction, Inspection, Alignment, Photometry, Light Curve, and Fitting;
- coordinates as the canonical target identity, with target name optional;
- HOPS-style target-name lookup through ExoClock/SIMBAD, with offline NASA and reusable cache fallbacks;
- automatic FITS frame classification with explicit calibration waivers;
- background processing, safe cancellation, checkpoints, and continuous project autosave;
- real reduced-FITS viewing with working pan, zoom, invert, reset, and target/comparison overlays;
- optional plate solving that validates an existing FITS WCS first, then uses bounded Gaia attempts with a visible timeline and manual target placement fallback;
- the original HOPS star detection, Gaussian fitting, geometric-centering option, variable aperture, sky-annulus, and differential-photometry calculations behind the new interface;
- a required Light Curve review where anomalous comparison stars can be excluded before HOPS-compatible fitting and export;
- target-coordinate-driven planet defaults, automatic HOPS passband/exposure detection, a rendered Preview Fit gate, and transactional full-fit outputs;
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

LEAPS v1 is not a broader astronomy suite. Linux installers, Intel Mac builds, blind or secondary solvers, telemetry, and a user-facing CLI are intentionally out of scope. The observing planner remains available under Tools.

This fork retains upstream HOPS scientific code and licensing notices. See [LICENSE](LICENSE).
