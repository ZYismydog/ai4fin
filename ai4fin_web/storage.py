"""Durable, isolated filesystem storage for web research runs."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from .models import RunSummary, StageDetail

logger = logging.getLogger(__name__)

_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{7,63}$")
_STAGE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
_RUN_RECORD = "run.json"


class RunNotFound(LookupError):
    """Raised when a run ID has no durable record."""


class ReportUnavailable(LookupError):
    """Raised when a run does not have a final report yet."""


class RunStore:
    """Persist summaries and artifacts below one isolated directory per run."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run(self, summary: RunSummary) -> RunSummary:
        # Defensive: the root may have been removed while the process is
        # running (e.g. temp cleanup); recreate it before writing.
        self.root.mkdir(parents=True, exist_ok=True)
        run_dir = self._run_path(summary.id)
        try:
            run_dir.mkdir()
        except FileExistsError:
            raise FileExistsError(f"Run already exists: {summary.id}") from None
        for name in ("uploads", "materials", "stages", "reports"):
            (run_dir / name).mkdir()
        self._write_model(run_dir / _RUN_RECORD, summary)
        return summary

    def update_summary(self, summary: RunSummary) -> None:
        run_dir = self._existing_run_path(summary.id)
        self._write_model(run_dir / _RUN_RECORD, summary)

    def get_summary(self, run_id: str) -> RunSummary:
        record = self._existing_run_path(run_id) / _RUN_RECORD
        try:
            return RunSummary.model_validate_json(record.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise RunNotFound(f"Run not found: {run_id}") from None

    def list_summaries(self, *, include_archived: bool = False) -> list[RunSummary]:
        self.root.mkdir(parents=True, exist_ok=True)
        summaries: list[RunSummary] = []
        for run_dir in self.root.iterdir():
            if not run_dir.is_dir() or not _RUN_ID_PATTERN.fullmatch(run_dir.name):
                continue
            try:
                summary = RunSummary.model_validate_json(
                    (run_dir / _RUN_RECORD).read_text(encoding="utf-8")
                )
            except (FileNotFoundError, OSError, UnicodeError, ValidationError, ValueError):
                logger.warning("Skipping malformed web run record: %s", run_dir.name)
                continue
            if summary.archived and not include_archived:
                continue
            summaries.append(summary)
        return sorted(summaries, key=lambda item: item.created_at, reverse=True)

    def set_archived(self, run_id: str, archived: bool) -> RunSummary:
        """Toggle the archived flag on a run's record."""
        path = self._existing_run_path(run_id) / _RUN_RECORD
        try:
            summary = RunSummary.model_validate_json(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, ValidationError, ValueError) as exc:
            raise RunNotFound(f"Run not found: {run_id}") from exc
        summary.archived = bool(archived)
        self._atomic_write_text(path, summary.model_dump_json(indent=2))
        return summary

    def delete_run(self, run_id: str) -> None:
        """Permanently delete a run's directory and all its artifacts."""
        import shutil

        run_dir = self._existing_run_path(run_id)
        shutil.rmtree(run_dir, ignore_errors=True)

    def save_upload(self, run_id: str, filename: str, content: bytes) -> Path:
        safe_name = self._safe_filename(filename)
        destination = self.upload_directory(run_id) / safe_name
        if destination.exists():
            raise FileExistsError(f"Upload already exists: {safe_name}")
        destination.write_bytes(content)
        return destination

    def upload_path(self, run_id: str, filename: str) -> Path:
        return self.upload_directory(run_id) / self._safe_filename(filename)

    def upload_directory(self, run_id: str) -> Path:
        return self._existing_run_path(run_id) / "uploads"

    def material_directory(self, run_id: str) -> Path:
        return self._existing_run_path(run_id) / "materials"

    def write_stage_detail(self, run_id: str, detail: StageDetail) -> None:
        stage_id = self._validate_stage_id(detail.id)
        path = self._existing_run_path(run_id) / "stages" / f"{stage_id}.json"
        self._write_model(path, detail)

    def read_stage_detail(self, run_id: str, stage_id: str) -> StageDetail:
        valid_stage_id = self._validate_stage_id(stage_id)
        path = self._existing_run_path(run_id) / "stages" / f"{valid_stage_id}.json"
        try:
            return StageDetail.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return StageDetail(
                id=valid_stage_id,
                title=valid_stage_id,
                status="pending",
                available=False,
                summary="该环节没有可用产出。",
            )

    def write_report(self, run_id: str, markdown: str) -> None:
        path = self._existing_run_path(run_id) / "reports" / "report.md"
        self._atomic_write_text(path, markdown)

    def read_report(self, run_id: str) -> str:
        path = self._existing_run_path(run_id) / "reports" / "report.md"
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise ReportUnavailable(f"Report unavailable for run: {run_id}") from None

    def reports_directory(self, run_id: str) -> Path:
        """The run's report-tree directory (holds the write_report_tree output)."""
        return self._existing_run_path(run_id) / "reports"

    def read_stage_artifact(self, run_id: str, stage_id: str) -> str | None:
        """Full agent output for one stage from the report tree, if present."""
        from .report_tree import read_stage_report

        reports_dir = self.reports_directory(run_id)
        return read_stage_report(reports_dir, stage_id)

    def read_final_review(self, run_id: str) -> str | None:
        """The user-facing final review section, if the report tree has one."""
        from .report_tree import read_final_review

        return read_final_review(self.reports_directory(run_id))

    def read_financial_analysis_pack(self, run_id: str) -> str | None:
        """The annual-report financial analysis pack (evidence lookup output),
        if the run produced one at reports/1_analysts/financial_analysis_pack.md."""
        path = self.reports_directory(run_id) / "1_analysts" / "financial_analysis_pack.md"
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    def read_chat_messages(self, run_id: str) -> list[dict]:
        """Persisted Q&A chat messages for a run (empty when never asked)."""
        path = self._run_path(run_id) / "chat_history.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, ValueError):
            return []

    def append_chat_messages(self, run_id: str, messages: list[dict]) -> None:
        """Append Q&A messages to the run's persisted chat history."""
        path = self._run_path(run_id) / "chat_history.json"
        existing = self.read_chat_messages(run_id)
        existing.extend(messages)
        path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def report_package_path(self, run_id: str) -> Path:
        """Path to report_package.json for Report Q&A, or raises ReportUnavailable."""
        path = self.reports_directory(run_id) / "report_package.json"
        if not path.is_file():
            raise ReportUnavailable(
                f"该任务未完成，报告包不可用（run: {run_id}），无法继续问答"
            )
        return path

    def recover_interrupted_runs(self) -> None:
        for summary in self.list_summaries():
            if summary.status in {"queued", "running"}:
                self.update_summary(
                    summary.model_copy(
                        update={"status": "interrupted", "current_stage_id": None}
                    )
                )

    def _run_path(self, run_id: str) -> Path:
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise RunNotFound(f"Invalid run ID: {run_id}")
        return self.root / run_id

    def _existing_run_path(self, run_id: str) -> Path:
        run_dir = self._run_path(run_id)
        if not (run_dir / _RUN_RECORD).is_file():
            raise RunNotFound(f"Run not found: {run_id}")
        return run_dir

    @staticmethod
    def _validate_stage_id(stage_id: str) -> str:
        if not _STAGE_ID_PATTERN.fullmatch(stage_id):
            raise ValueError(f"Invalid stage ID: {stage_id}")
        return stage_id

    @staticmethod
    def _safe_filename(filename: str) -> str:
        safe_name = Path(filename).name
        if safe_name in {"", ".", ".."}:
            raise ValueError("Upload filename is empty or unsafe")
        return safe_name

    @classmethod
    def _write_model(cls, path: Path, model: RunSummary | StageDetail) -> None:
        payload = json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2)
        cls._atomic_write_text(path, payload + "\n")

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(content, encoding="utf-8")
            for attempt in range(5):
                try:
                    temporary.replace(path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.01 * (attempt + 1))
        finally:
            temporary.unlink(missing_ok=True)
