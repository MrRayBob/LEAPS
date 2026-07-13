from __future__ import annotations

import json
from pathlib import Path

import pytest

from leaps.models import LEAPSError, ProjectManifest, StageID, StageStatus
from leaps.project import ProjectWorkspace


def test_manifest_round_trip_uses_versioned_plain_values(tmp_path: Path) -> None:
    path = tmp_path / "project.json"
    manifest = ProjectManifest(name="Test transit")
    manifest.stages[StageID.DATA_TARGET.value].status = StageStatus.COMPLETE
    manifest.save(path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 2
    assert payload["stages"]["data_target"]["status"] == "complete"
    restored = ProjectManifest.load(path)
    assert restored.stages["data_target"].status is StageStatus.COMPLETE


def test_project_paths_remain_relative_and_unlock_next_stage(tmp_path: Path) -> None:
    raw = tmp_path / "science" / "light_001.fits"
    raw.parent.mkdir()
    raw.touch()
    project = ProjectWorkspace.create(tmp_path, "Portable run")
    project.manifest.raw_files["science"] = [project.relative(raw)]
    project.set_stage(StageID.DATA_TARGET, StageStatus.COMPLETE, "Target selected")
    assert project.manifest.raw_files["science"] == ["science/light_001.fits"]
    assert project.resolve("science/light_001.fits") == raw
    assert project.manifest.stages[StageID.REDUCTION.value].status is StageStatus.READY
    assert project.workspace == tmp_path / "LEAPS"
    assert project.manifest_path == tmp_path / "LEAPS" / "project.json"


def test_transaction_replaces_output_only_after_commit(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path)
    target = project.outputs_dir / StageID.REDUCTION.value
    target.mkdir()
    (target / "old.txt").write_text("last success")
    pending, resolved_target = project.begin_transaction(StageID.REDUCTION)
    (pending / "new.txt").write_text("new success")
    assert (target / "old.txt").read_text() == "last success"
    project.commit_transaction(pending, resolved_target)
    assert not (target / "old.txt").exists()
    assert (target / "new.txt").read_text() == "new success"


def test_legacy_hidden_workspace_migrates_with_outputs_and_relative_references(tmp_path: Path) -> None:
    legacy = tmp_path / ".leaps"
    manifest = ProjectManifest(name="Legacy transit")
    state = manifest.stages[StageID.REDUCTION.value]
    state.checkpoint = ".leaps/checkpoints/reduction.json"
    state.output_path = ".leaps/outputs/reduction"
    manifest.save(legacy / "project.json")
    output = legacy / "outputs" / "reduction" / "reduced.fits"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"generated")
    log = legacy / "logs" / "leaps.jsonl"
    log.parent.mkdir()
    log.write_text('{"event":"saved"}\n', encoding="utf-8")
    checkpoint = legacy / "checkpoints" / "reduction.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text('{"frame":12}', encoding="utf-8")

    project = ProjectWorkspace.open(tmp_path)

    assert project.workspace == tmp_path / "LEAPS"
    assert not legacy.exists()
    assert (project.outputs_dir / "reduction" / "reduced.fits").read_bytes() == b"generated"
    assert (project.logs_dir / "leaps.jsonl").read_text(encoding="utf-8") == '{"event":"saved"}\n'
    assert (project.checkpoints_dir / "reduction.json").read_text(encoding="utf-8") == '{"frame":12}'
    state = project.manifest.stages[StageID.REDUCTION.value]
    assert state.checkpoint == "LEAPS/checkpoints/reduction.json"
    assert state.output_path == "LEAPS/outputs/reduction"


def test_project_open_stops_when_visible_and_legacy_workspaces_both_exist(tmp_path: Path) -> None:
    ProjectManifest(name="Visible").save(tmp_path / "LEAPS" / "project.json")
    ProjectManifest(name="Legacy").save(tmp_path / ".leaps" / "project.json")

    with pytest.raises(LEAPSError) as error:
        ProjectWorkspace.open(tmp_path)

    assert error.value.code == "PROJECT_WORKSPACE_CONFLICT"
    assert (tmp_path / "LEAPS" / "project.json").exists()
    assert (tmp_path / ".leaps" / "project.json").exists()


def test_failed_legacy_manifest_update_rolls_folder_back_without_data_loss(
    tmp_path: Path, monkeypatch
) -> None:
    legacy = tmp_path / ".leaps"
    manifest = ProjectManifest(name="Legacy transit")
    manifest.stages[StageID.REDUCTION.value].output_path = ".leaps/outputs/reduction"
    manifest.save(legacy / "project.json")
    output = legacy / "outputs" / "reduction" / "reduced.fits"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"generated")

    def fail_save(_manifest, _path):
        raise OSError("project manifest became unwritable")

    monkeypatch.setattr(ProjectManifest, "save", fail_save)
    with pytest.raises(LEAPSError) as error:
        ProjectWorkspace.open(tmp_path)

    assert error.value.code == "PROJECT_MIGRATION_FAILED"
    assert legacy.exists()
    assert not (tmp_path / "LEAPS").exists()
    assert output.read_bytes() == b"generated"
    saved = json.loads((legacy / "project.json").read_text(encoding="utf-8"))
    assert saved["stages"]["reduction"]["output_path"] == ".leaps/outputs/reduction"


def test_project_creation_does_not_overwrite_unrelated_visible_folder(tmp_path: Path) -> None:
    unrelated = tmp_path / "LEAPS"
    unrelated.mkdir()
    (unrelated / "notes.txt").write_text("not a project")

    with pytest.raises(LEAPSError) as error:
        ProjectWorkspace.create(tmp_path)

    assert error.value.code == "PROJECT_WORKSPACE_CONFLICT"
    assert (unrelated / "notes.txt").read_text() == "not a project"


def test_project_open_stops_when_valid_workspace_contains_unrelated_data(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path, "Visible")
    note = project.workspace / "observer-notes.txt"
    note.write_text("keep me", encoding="utf-8")

    with pytest.raises(LEAPSError) as error:
        ProjectWorkspace.open(tmp_path)

    assert error.value.code == "PROJECT_WORKSPACE_CONFLICT"
    assert note.read_text(encoding="utf-8") == "keep me"


def test_reset_deletes_only_generated_workspace_and_preserves_raw_frames(tmp_path: Path) -> None:
    raw = tmp_path / "light_001.fits"
    raw.write_bytes(b"raw pixels")
    project = ProjectWorkspace.create(tmp_path, "Safe reset")
    generated = project.outputs_dir / "reduction" / "reduced.fits"
    generated.parent.mkdir()
    generated.write_bytes(b"generated pixels")
    project.manifest.raw_files["science"] = [raw.name]
    project.save()

    removed = project.delete_generated_data()

    assert removed >= len(b"generated pixels")
    assert raw.read_bytes() == b"raw pixels"
    assert not project.workspace.exists()
    assert list(tmp_path.iterdir()) == [raw]


def test_reset_rejects_workspace_symlink_without_touching_target(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path, "Linked project")
    real_workspace = tmp_path / "workspace-backup"
    project.workspace.rename(real_workspace)
    project.workspace.symlink_to(real_workspace, target_is_directory=True)

    with pytest.raises(LEAPSError) as error:
        project.delete_generated_data()

    assert error.value.code == "PROJECT_RESET_SYMLINK"
    assert (real_workspace / "project.json").exists()


def test_interrupted_reset_reports_remaining_staging_folder(tmp_path: Path, monkeypatch) -> None:
    project = ProjectWorkspace.create(tmp_path, "Interrupted reset")

    def fail_delete(_path):
        raise OSError("disk became unavailable")

    monkeypatch.setattr("leaps.project.shutil.rmtree", fail_delete)
    with pytest.raises(LEAPSError) as error:
        project.delete_generated_data()

    assert error.value.code == "PROJECT_RESET_INCOMPLETE"
    assert not project.workspace.exists()
    assert len(list(tmp_path.glob(".LEAPS-reset-*"))) == 1


def test_reset_rejects_any_path_outside_observing_run(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path, "Unsafe path")
    project.workspace = tmp_path.parent

    with pytest.raises(LEAPSError) as error:
        project.delete_generated_data()

    assert error.value.code == "PROJECT_RESET_UNSAFE_PATH"
    assert (tmp_path / "LEAPS" / "project.json").exists()
