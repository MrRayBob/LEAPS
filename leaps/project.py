from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .models import LEAPSError, ProjectManifest, StageID, StageState, StageStatus, utc_now


class ProjectWorkspace:
    WORKSPACE_NAME = "LEAPS"
    LEGACY_WORKSPACE_NAME = ".leaps"
    GENERATED_TOP_LEVEL_ENTRIES = {
        "project.json",
        "project.json.tmp",
        "logs",
        "cache",
        "checkpoints",
        "outputs",
        "tmp",
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",
    }

    def __init__(self, root: Path, manifest: ProjectManifest) -> None:
        self.root = root.resolve()
        self.workspace = self.root / self.WORKSPACE_NAME
        self.manifest_path = self.workspace / "project.json"
        self.manifest = manifest
        for directory in (
            self.logs_dir,
            self.cache_dir,
            self.checkpoints_dir,
            self.outputs_dir,
            self.temporary_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def logs_dir(self) -> Path:
        return self.workspace / "logs"

    @property
    def cache_dir(self) -> Path:
        return self.workspace / "cache"

    @property
    def checkpoints_dir(self) -> Path:
        return self.workspace / "checkpoints"

    @property
    def outputs_dir(self) -> Path:
        return self.workspace / "outputs"

    @property
    def temporary_dir(self) -> Path:
        return self.workspace / "tmp"

    @classmethod
    def create(cls, root: str | Path, name: str | None = None) -> ProjectWorkspace:
        root_path = Path(root).expanduser().resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        if cls.has_workspace(root_path):
            raise cls._workspace_conflict(
                root_path,
                "A LEAPS workspace already exists in this observing run.",
            )
        manifest = ProjectManifest(name=name or root_path.name)
        project = cls(root_path, manifest)
        project.save()
        return project

    @classmethod
    def open(cls, root: str | Path) -> ProjectWorkspace:
        root_path = Path(root).expanduser().resolve()
        workspace = cls._existing_workspace(root_path)
        if workspace is not None:
            manifest = cls._load_manifest(workspace / "project.json")
            if workspace.name == cls.LEGACY_WORKSPACE_NAME:
                workspace = cls._migrate_legacy_workspace(root_path, workspace)
                try:
                    project = cls(root_path, manifest)
                    project._rewrite_legacy_references()
                    return project
                except Exception as exc:
                    rollback_error: OSError | None = None
                    legacy = root_path / cls.LEGACY_WORKSPACE_NAME
                    try:
                        if not os.path.lexists(legacy):
                            workspace.rename(legacy)
                    except OSError as rollback_exc:
                        rollback_error = rollback_exc
                    details = str(exc)
                    if rollback_error is not None:
                        details += f"\nRollback failed: {rollback_error}"
                    raise LEAPSError(
                        "PROJECT_MIGRATION_FAILED",
                        "The legacy project could not be migrated safely",
                        (
                            "LEAPS returned the project to .leaps/."
                            if rollback_error is None
                            else "The project files remain intact, but the folder name needs review."
                        ),
                        [
                            "Check folder permissions",
                            "Close other applications using the project",
                            f"Review {root_path}",
                            "Try again",
                        ],
                        stage=StageID.DATA_TARGET,
                        technical_details=details,
                    ) from exc
            return cls(root_path, manifest)
        return cls.import_hops(root_path)

    @classmethod
    def has_workspace(cls, root: str | Path) -> bool:
        root_path = Path(root).expanduser().resolve()
        return any(
            os.path.lexists(root_path / name)
            for name in (cls.WORKSPACE_NAME, cls.LEGACY_WORKSPACE_NAME)
        )

    @classmethod
    def has_project(cls, root: str | Path) -> bool:
        root_path = Path(root).expanduser().resolve()
        return any(
            (root_path / name / "project.json").is_file()
            for name in (cls.WORKSPACE_NAME, cls.LEGACY_WORKSPACE_NAME)
        )

    @classmethod
    def _existing_workspace(cls, root: Path) -> Path | None:
        visible = root / cls.WORKSPACE_NAME
        legacy = root / cls.LEGACY_WORKSPACE_NAME
        visible_exists = os.path.lexists(visible)
        legacy_exists = os.path.lexists(legacy)
        if visible_exists and legacy_exists:
            raise cls._workspace_conflict(
                root,
                "Both LEAPS/ and the legacy .leaps/ folder are present.",
            )
        workspace = visible if visible_exists else legacy if legacy_exists else None
        if workspace is None:
            return None
        if workspace.is_symlink():
            raise cls._workspace_conflict(
                root,
                f"{workspace.name}/ is a symbolic link and cannot be used safely.",
            )
        manifest = workspace / "project.json"
        if not workspace.is_dir() or not manifest.is_file() or manifest.is_symlink():
            raise cls._workspace_conflict(
                root,
                f"{workspace.name}/ exists but is not a valid LEAPS project folder.",
            )
        try:
            unrelated = sorted(
                child.name
                for child in workspace.iterdir()
                if child.name not in cls.GENERATED_TOP_LEVEL_ENTRIES
            )
        except OSError as exc:
            raise LEAPSError(
                "PROJECT_WORKSPACE_UNREADABLE",
                "The project folder could not be inspected",
                f"LEAPS could not safely read {workspace}.",
                ["Check folder permissions", "Close other applications using the folder", "Try again"],
                stage=StageID.DATA_TARGET,
                technical_details=str(exc),
            ) from exc
        if unrelated:
            names = ", ".join(unrelated[:3])
            remainder = len(unrelated) - 3
            if remainder > 0:
                names += f", and {remainder} more"
            raise cls._workspace_conflict(
                root,
                f"{workspace.name}/ contains files LEAPS did not create: {names}.",
            )
        return workspace

    @classmethod
    def _migrate_legacy_workspace(cls, root: Path, legacy: Path) -> Path:
        visible = root / cls.WORKSPACE_NAME
        if os.path.lexists(visible):
            raise cls._workspace_conflict(
                root,
                "LEAPS/ already exists, so the legacy project was not moved.",
            )
        try:
            legacy.rename(visible)
        except OSError as exc:
            raise LEAPSError(
                "PROJECT_MIGRATION_FAILED",
                "The legacy project could not be moved",
                "LEAPS left the existing .leaps folder unchanged.",
                ["Check folder permissions", "Close other applications using the project", "Try again"],
                stage=StageID.DATA_TARGET,
                technical_details=str(exc),
            ) from exc
        return visible

    @staticmethod
    def _load_manifest(path: Path) -> ProjectManifest:
        try:
            return ProjectManifest.load(path)
        except Exception as exc:
            raise LEAPSError(
                "PROJECT_MANIFEST_INVALID",
                "The LEAPS project information is damaged",
                f"The project manifest at {path} could not be read.",
                ["Restore project.json from a backup", "Export diagnostics", "Reset the project data"],
                stage=StageID.DATA_TARGET,
                technical_details=str(exc),
            ) from exc

    @classmethod
    def _workspace_conflict(cls, root: Path, message: str) -> LEAPSError:
        return LEAPSError(
            "PROJECT_WORKSPACE_CONFLICT",
            "The project folder needs attention",
            message,
            [
                f"Review {root / cls.WORKSPACE_NAME}",
                "Rename unrelated files or keep only one LEAPS project folder",
                "Try opening the observing run again",
            ],
            stage=StageID.DATA_TARGET,
        )

    @classmethod
    def import_hops(cls, root: Path) -> ProjectWorkspace:
        project = cls.create(root, root.name)
        legacy_log = root / "log.yaml"
        if legacy_log.exists():
            try:
                import yaml

                values = yaml.safe_load(legacy_log.read_text(encoding="utf-8")) or {}
                project.manifest.target_name = str(values.get("target_name", ""))
                coordinates = str(values.get("target_ra_dec", "")).split()
                if len(coordinates) == 2:
                    project.manifest.target_ra, project.manifest.target_dec = coordinates
                for stage in StageID:
                    if values.get(f"{stage.value}_complete"):
                        project.manifest.stages[stage.value] = StageState(
                            status=StageStatus.COMPLETE,
                            summary="Imported from HOPS",
                        )
                project.manifest.warnings.append(
                    {
                        "code": "IMPORTED_HOPS",
                        "message": "Legacy HOPS state was imported without changing its files.",
                    }
                )
                project.save()
            except Exception as exc:
                project.manifest.warnings.append({"code": "HOPS_IMPORT_PARTIAL", "message": str(exc)})
                project.save()
        return project

    def save(self) -> None:
        self.manifest.save(self.manifest_path)

    def _rewrite_legacy_references(self) -> None:
        prefix = f"{self.LEGACY_WORKSPACE_NAME}/"
        replacement = f"{self.WORKSPACE_NAME}/"
        changed = False
        for state in self.manifest.stages.values():
            for attribute in ("checkpoint", "output_path"):
                value = getattr(state, attribute)
                if isinstance(value, str) and value.startswith(prefix):
                    setattr(state, attribute, replacement + value[len(prefix) :])
                    changed = True
        if changed:
            self.save()

    def workspace_size(self) -> int:
        total = 0
        if not self.workspace.exists() or self.workspace.is_symlink():
            return total
        for directory, subdirectories, filenames in os.walk(self.workspace, followlinks=False):
            base = Path(directory)
            subdirectories[:] = [
                name for name in subdirectories if not (base / name).is_symlink()
            ]
            for name in filenames:
                try:
                    total += (base / name).lstat().st_size
                except OSError:
                    continue
        return total

    def delete_generated_data(self) -> int:
        """Delete only this validated LEAPS workspace, never the observing-run root."""
        workspace = self.workspace
        if workspace.parent != self.root or workspace.name not in {
            self.WORKSPACE_NAME,
            self.LEGACY_WORKSPACE_NAME,
        }:
            raise LEAPSError(
                "PROJECT_RESET_UNSAFE_PATH",
                "Project reset was stopped",
                "The generated-data folder is not a direct LEAPS workspace beside the FITS data.",
                ["Open the observing run again", "Export diagnostics"],
                stage=StageID.DATA_TARGET,
                technical_details=str(workspace),
            )
        if workspace.is_symlink() or self.manifest_path.is_symlink():
            raise LEAPSError(
                "PROJECT_RESET_SYMLINK",
                "Project reset was stopped",
                "LEAPS will not delete a workspace or manifest reached through a symbolic link.",
                ["Replace the link with a normal project folder", "Remove it manually after inspection"],
                stage=StageID.DATA_TARGET,
            )
        if not workspace.exists():
            return 0
        on_disk = self._load_manifest(self.manifest_path)
        if on_disk.project_id != self.manifest.project_id:
            raise LEAPSError(
                "PROJECT_RESET_ID_MISMATCH",
                "Project reset was stopped",
                "The project on disk no longer matches the project open in LEAPS.",
                ["Close and reopen the observing run", "Try reset again"],
                stage=StageID.DATA_TARGET,
            )
        removed_bytes = self.workspace_size()
        staging = self.root / f".LEAPS-reset-{uuid.uuid4().hex[:10]}"
        try:
            workspace.rename(staging)
        except OSError as exc:
            raise LEAPSError(
                "PROJECT_RESET_FAILED",
                "Project data could not be reset",
                "LEAPS did not remove any project files.",
                ["Check folder permissions", "Close other applications using the folder", "Try again"],
                stage=StageID.DATA_TARGET,
                technical_details=str(exc),
            ) from exc
        try:
            shutil.rmtree(staging)
        except OSError as exc:
            raise LEAPSError(
                "PROJECT_RESET_INCOMPLETE",
                "Project reset needs attention",
                f"The active project was removed, but some generated data remains at {staging}.",
                ["Delete the remaining reset folder manually", "Verify available disk access"],
                stage=StageID.DATA_TARGET,
                technical_details=str(exc),
            ) from exc
        return removed_bytes

    def relative(self, path: str | Path) -> str:
        resolved = Path(path).expanduser().resolve()
        try:
            return resolved.relative_to(self.root).as_posix()
        except ValueError:
            return resolved.as_posix()

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (self.root / candidate).resolve()

    def set_stage(
        self,
        stage: StageID,
        status: StageStatus,
        summary: str,
        *,
        progress: float | None = None,
        checkpoint: str | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        state = self.manifest.stages[stage.value]
        state.status = status
        state.summary = summary
        state.updated_at = utc_now()
        if progress is not None:
            state.progress = progress
        if checkpoint is not None:
            state.checkpoint = checkpoint
        if output_path is not None:
            state.output_path = self.relative(output_path)
        if status == StageStatus.COMPLETE:
            stages = list(StageID)
            next_index = stages.index(stage) + 1
            if next_index < len(stages):
                next_stage = self.manifest.stages[stages[next_index].value]
                if next_stage.status == StageStatus.LOCKED:
                    next_stage.status = StageStatus.READY
                    next_stage.summary = "Ready"
        self.save()

    def begin_transaction(self, stage: StageID) -> tuple[Path, Path]:
        target = self.outputs_dir / stage.value
        pending = self.temporary_dir / f"{stage.value}-pending"
        if pending.exists():
            shutil.rmtree(pending)
        pending.mkdir(parents=True)
        return pending, target

    def discard_pending_transaction(self, stage: StageID) -> bool:
        """Remove only the recognized temporary output for one stage."""
        pending = self.temporary_dir / f"{stage.value}-pending"
        if pending.is_symlink():
            pending.unlink()
            return True
        if pending.exists():
            shutil.rmtree(pending)
            return True
        return False

    def commit_transaction(self, pending: Path, target: Path) -> None:
        previous = target.with_name(target.name + "-previous")
        if previous.exists():
            shutil.rmtree(previous)
        if target.exists():
            target.replace(previous)
        pending.replace(target)
        if previous.exists():
            shutil.rmtree(previous)

    def update_settings(self, **values: Any) -> None:
        self.manifest.settings.update(values)
        self.save()
