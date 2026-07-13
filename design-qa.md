# LEAPS design QA

## Visual target

- Approved source: `docs/design/leaps-plate-solve-reference.png`
- Implemented capture: `docs/design/leaps-plate-solve-implementation.png`
- Side-by-side comparison: `docs/design/reference-vs-implementation.png`
- Calibration-waiver spacing comparison: `docs/design/waiver-checkbox-spacing-comparison.png`
- Reference viewport: 1487 × 1058, including 40 px of generated-image canvas below the app window
- Implemented app viewport: 1487 × 1018
- State: Photometry active; Gaia HTTP 503 failure after three bounded attempts; retry, manual target placement, and diagnostic-copy actions visible

## Review evidence

- The source and implementation were placed together in a single comparison image at equal width.
- The implementation preserves the approved three-column proportions, 265 px workflow sidebar, FITS workspace, 390 px recovery inspector, observatory palette, cyan/amber state language, target and detected-star overlays, metadata strip, and global autosave footer.
- The header and image area were adjusted after the first render to match the source proportions: 114 px page header, 68 px session footer, and approximately 706 px visible FITS image height.
- Child-frame stylesheet leakage was removed after comparison so active-navigation and target-information labels no longer gained unintended cyan or divider borders.
- The three calibration-waiver checkboxes now use a dedicated vertical layout with 10 px row spacing and 6 px top breathing room; the focused before/after capture confirms that labels and indicators no longer crowd adjacent rows.
- The production UI intentionally uses native macOS window controls rather than recreating traffic-light controls. The Observing Planner is also present under Tools as required by the product plan.
- Primary actions, sidebar navigation, manual target placement, copy diagnostics, offline-data controls, first-run settings, project setup, background processing, comparison-star approval, fitting actions, and exports are connected to application behavior.

## Accessibility and responsive checks

- The main window resizes down to 1120 × 720 and restores valid saved geometry.
- Scientific controls and consequential actions use keyboard-focusable controls with descriptive tooltips.
- Circled information controls expose an accessible name and non-empty tooltip; automated UI coverage verifies at least 15 are present in the selected workspace build.
- Long recovery text wraps in a scrollable inspector, and processing never disables the entire application window.

## Verification

- `python3 -m py_compile leaps/*.py leaps/ui/*.py hops/hops_tools/*.py hops/thirdparty/twirl/*.py`
- `QT_QPA_PLATFORM=offscreen pytest -q`: 17 passed
- `pyside6-deploy -c pysidedeploy.spec --dry-run --force`: deployment command generated successfully
- Offscreen Qt launch and deterministic screenshot: passed
- Reference versus implementation visual review: passed

final result: passed
