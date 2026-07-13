from __future__ import annotations

from leaps.fits_inventory import FrameRecord
from leaps.models import StageID
from leaps.ui.main_window import MainWindow
from leaps.ui.pages import DataTargetPage
from leaps.ui.widgets import InfoButton


def test_demo_window_opens_selected_plate_solve_state(qapp) -> None:
    window = MainWindow(demo=True)
    window.show()
    qapp.processEvents()
    assert window.stack.currentWidget() is window.plate_page
    assert window.stage_buttons[StageID.PHOTOMETRY].active
    assert window.plate_page.inspector.retry.isEnabled()
    window.close()


def test_scientific_controls_have_accessible_information_buttons(qapp) -> None:
    window = MainWindow(demo=True)
    info_buttons = window.findChildren(InfoButton)
    assert len(info_buttons) >= 15
    assert all(button.toolTip().strip() for button in info_buttons)
    assert all(button.accessibleName() == "Information" for button in info_buttons)
    window.close()


def test_frame_assignment_cards_use_live_filename_classifiers(qapp) -> None:
    page = DataTargetPage()
    records = [
        FrameRecord("bias_001.fits", "bias", 1.0, "", (20, 20), 16, 0.0, "a"),
        FrameRecord("d_001.fits", "dark", 1.0, "", (20, 20), 16, 30.0, "b"),
        FrameRecord("flat_001.fits", "flat", 1.0, "", (20, 20), 16, 5.0, "c"),
        FrameRecord("image_001.fits", "science", 1.0, "", (20, 20), 16, 30.0, "d"),
    ]
    page.set_records(records)
    assert page.assignment_cards["bias"].count.text() == "1 selected"
    assert page.assignment_cards["dark"].count.text() == "0 selected"
    assert page.assignment_cards["flat"].count.text() == "1 selected"
    assert page.assignment_cards["science"].count.text() == "1 selected"
    assert page.counts.text() == "3 assigned · 1 unmatched"

    page.assignment_cards["dark"].classifier.setText("Dark, D")
    qapp.processEvents()
    assert page.assignment_cards["dark"].count.text() == "1 selected"
    assert page.counts.text() == "4 assigned · 0 unmatched"


def test_frame_assignment_counts_filename_matches_before_header_scan(qapp, tmp_path) -> None:
    for index in range(21):
        (tmp_path / f"bias_{index + 1:03d}.fits").touch()
    for index in range(7):
        (tmp_path / f"dark_60_{index + 1:03d}.fits").touch()
    for index in range(5):
        (tmp_path / f"flat_{index + 1:03d}.fits").touch()
    for index in range(12):
        (tmp_path / f"TrES-3_Cousins_R_{index + 1:03d}.fits").touch()

    page = DataTargetPage()
    page.preview_folder(tmp_path)
    assert page.assignment_cards["bias"].count.text() == "21 selected"
    assert page.assignment_cards["dark"].count.text() == "7 selected"
    assert page.assignment_cards["flat"].count.text() == "5 selected"
    assert page.assignment_cards["science"].count.text() == "0 selected"
    assert page.counts.text() == "33 assigned · 12 unmatched"

    page.assignment_cards["science"].classifier.setText("TrES-3")
    qapp.processEvents()
    assert page.assignment_cards["science"].count.text() == "12 selected"
    assert page.counts.text() == "45 assigned · 0 unmatched"
