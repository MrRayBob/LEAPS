# LEAPS design QA

## Visual target

- Approved source: `docs/design/leaps-plate-solve-reference.png`
- Implemented capture: `docs/design/leaps-plate-solve-implementation.png`
- Side-by-side comparison: `docs/design/reference-vs-implementation.png`
- Calibration-waiver spacing comparison: `docs/design/waiver-checkbox-spacing-comparison.png`
- Target-name resolution comparison: `docs/design/target-name-resolution-comparison.png`
- Locked-sidebar reference: `docs/design/leaps-sidebar-lock-reference.png`
- Locked-sidebar implementation: `docs/design/leaps-sidebar-lock-implementation.png`
- Locked-sidebar comparison: `docs/design/leaps-sidebar-lock-comparison.png`
- Reference viewport: 1487 × 1058, including 40 px of generated-image canvas below the app window
- Implemented app viewport: 1487 × 1018
- Reference state: Photometry active; Gaia HTTP 503 failure after three bounded attempts; retry, manual target placement, and diagnostic-copy actions visible
- Production state: the current project's reduced FITS frame and target replace all illustrative WTS-2 b data; plate solving is optional and comparison stars are reviewed in the same workspace

## Review evidence

- The source and implementation were placed together in a single comparison image at equal width.
- The implementation preserves the approved three-column proportions, 265 px workflow sidebar, FITS workspace, 390 px recovery inspector, observatory palette, cyan/amber state language, target and detected-star overlays, metadata strip, and global autosave footer.
- The header and image area were adjusted after the first render to match the source proportions: 114 px page header, 68 px session footer, and approximately 706 px visible FITS image height.
- Child-frame stylesheet leakage was removed after comparison so active-navigation and target-information labels no longer gained unintended cyan or divider borders.
- The three calibration-waiver checkboxes now use a dedicated vertical layout with 10 px row spacing and 6 px top breathing room; the focused before/after capture confirms that labels and indicators no longer crowd adjacent rows.
- Target-name lookup preserves the existing coordinate-card structure, adds a compact inline status row, and fills RA/DEC in the familiar HOPS format after a successful ExoClock/SIMBAD or offline lookup.
- The locked workflow state now follows the supplied sidebar reference with a larger solid padlock, distinct muted-blue title and summary levels, and roomier icon alignment. Completed checkmarks and the active-stage ring remain unchanged so progress is still immediately recognizable.
- The production UI intentionally uses native macOS window controls rather than recreating traffic-light controls. The Observing Planner is also present under Tools as required by the product plan.
- Primary actions, sidebar navigation, manual target placement, copy diagnostics, offline-data controls, first-run settings, project setup, background processing, comparison-star approval, fitting actions, and exports are connected to application behavior.
- Pan, zoom, percentage zoom, invert, reset, and full-screen controls operate on a real FITS scene; aperture and sky-annulus overlays retain consistent line weight while the image is zoomed.
- The supplied TrES-3 project restores 21 bias, 7 dark, 5 flat, and 354 science assignments from its manifest instead of reverting to default filename classifiers.
- The TrES-3 header WCS is no longer trusted merely because it parses: its nominal coordinate landed on blank sky, so LEAPS corrected the pointing offset with 24 cached Gaia matches and snapped TrES-3 to the detected host at x 1029.5, y 932.5 (0.340 arcsec/pixel).
- Photometry uses HOPS star detection and measurement routines, preserves variable-aperture and geometric-centering controls, checkpoints every frame, and writes the expected aperture/Gaussian tables, light curves, FOV figures, and results figures.
- A real-data smoke run measured TrES-3 plus three ranked comparisons across 10 reduced frames; all 10 aperture light-curve rows were finite and the results figure was generated.

## Accessibility and responsive checks

- The main window resizes down to 1120 × 720 and restores valid saved geometry.
- Scientific controls and consequential actions use keyboard-focusable controls with descriptive tooltips.
- Circled information controls expose an accessible name and non-empty tooltip; automated UI coverage verifies at least 15 are present in the selected workspace build.
- Long recovery text wraps in a scrollable inspector, and processing never disables the entire application window.

## Verification

- `python3 -m py_compile leaps/*.py leaps/ui/*.py hops/hops_tools/*.py hops/thirdparty/twirl/*.py`
- `QT_QPA_PLATFORM=offscreen pytest -q`: 38 passed
- `.venv/bin/python -m ruff check .`: passed (the unchanged vendored `hops/` package is excluded)
- `pyside6-deploy -c pysidedeploy.spec --dry-run --force`: deployment command generated successfully
- Offscreen Qt launch and deterministic screenshot: passed
- Reference versus implementation visual review: passed
- Locked-sidebar source versus implementation review: passed; no P0, P1, or P2 visual issues found

## Fit preview file reveal

- Source visual: `/var/folders/wg/4c5ys1f50kg_hwllv6bpb4fc0000gn/T/TemporaryItems/NSIRD_screencaptureui_gM0bHN/Screenshot 2026-07-13 at 13.52.25.png`
- Implementation capture: `/private/tmp/leaps-fit-preview-view-in-files-2.png`
- Side-by-side comparison: `/private/tmp/leaps-fit-preview-comparison-2.png`
- The `View in Files` action sits directly beneath the preview and follows LEAPS button, spacing, icon, and color conventions.
- Generated fit previews are 2400×1680, and in-app rendering uses device-pixel-ratio-aware smooth scaling.
- The button remains disabled until a valid preview exists and reveals the exact image in Finder or Explorer.
- Focused fitting and UI tests, Ruff, compilation, and diff checks passed.

final result: passed
