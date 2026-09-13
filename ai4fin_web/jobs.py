"""Single-worker job queue for durable AI4Fin web runs."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import (
    CreateRunInput,
    RunEvent,
    RunSummary,
    StageDetail,
    StageSummary,
    UploadPayload,
)
from .projection import STAGES, ProjectionResult, project_state
from .runner import Ai4FinRunner
from .storage import RunStore

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"completed", "failed", "interrupted"}
_STOP = object()


class JobManager:
    """Persist and execute runs serially while publishing summary-only events."""

    def __init__(self, *, store: RunStore, runner: Ai4FinRunner) -> None:
        self._store = store
        self._runner = runner
        self._queue: queue.Queue[str | object] = queue.Queue()
        self._lock = threading.RLock()
        self._conditions: dict[str, threading.Condition] = {}
        self._versions: dict[str, int] = {}
        self._messages: dict[str, str] = {}
        self._requests: dict[str, CreateRunInput] = {}
        self._active_run_id: str | None = None
        self._closed = False
        self._store.recover_interrupted_runs()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="ai4fin-web-worker",
            daemon=True,
        )
        self._worker.start()

    def submit(
        self,
        request: CreateRunInput,
        uploads: Sequence[UploadPayload],
    ) -> RunSummary:
        with self._lock:
            if self._closed:
                raise RuntimeError("Job manager is closed")

        attachment_names = self._attachment_names(uploads)
        run_id = f"run-{uuid4().hex}"
        summary = RunSummary(
            id=run_id,
            subject=request.subject,
            analysis_date=request.analysis_date,
            query=request.query,
            status="queued",
            current_stage_id=None,
            progress=0,
            report_available=False,
            created_at=datetime.now(timezone.utc),
            stages=[
                StageSummary(id=stage.id, title=stage.title, status="pending")
                for stage in STAGES
            ],
            attachment_names=attachment_names,
            config=request.model_dump(
                exclude_none=True,
            ),
        )
        # Trim empty free-form fields so the snapshot only carries what the
        # browser actually submitted (mirrors the CLI selections).
        summary.config = {k: v for k, v in summary.config.items() if v != ""}
        self._store.create_run(summary)
        for upload, safe_name in zip(uploads, attachment_names, strict=True):
            self._store.save_upload(run_id, safe_name, upload.content)

        with self._lock:
            self._requests[run_id] = request
            self._conditions[run_id] = threading.Condition(self._lock)
            self._versions[run_id] = 0
            self._messages[run_id] = "任务已加入队列"
            self._queue.put(run_id)
        return summary

    def get(self, run_id: str) -> RunSummary:
        return self._store.get_summary(run_id)

    def list(self, *, include_archived: bool = False) -> list[RunSummary]:
        return self._store.list_summaries(include_archived=include_archived)

    def subscribe(self, run_id: str) -> Iterator[RunEvent]:
        self._store.get_summary(run_id)
        last_version = -1
        while True:
            with self._lock:
                condition = self._conditions.setdefault(
                    run_id, threading.Condition(self._lock)
                )
                condition.wait_for(
                    lambda last_version=last_version: self._versions.get(run_id, 0)
                    != last_version
                    or self._closed
                )
                version = self._versions.get(run_id, 0)
                if version == last_version and self._closed:
                    return
                last_version = version
                summary = self._store.get_summary(run_id)
                message = self._messages.get(run_id, "")

            yield self._event_from_summary(summary, message)
            if summary.status in _TERMINAL_STATUSES:
                return

    def read_stage(self, run_id: str, stage_id: str) -> StageDetail:
        return self._store.read_stage_detail(run_id, stage_id)

    def read_stage_artifact(self, run_id: str, stage_id: str) -> str | None:
        """Full agent output for a stage from the report tree (trace drawer)."""
        self._store.get_summary(run_id)
        return self._store.read_stage_artifact(run_id, stage_id)

    def read_final_review(self, run_id: str) -> str | None:
        """User-facing final review section from the report tree."""
        self._store.get_summary(run_id)
        return self._store.read_final_review(run_id)

    def read_financial_analysis_pack(self, run_id: str) -> str | None:
        """Annual-report financial analysis pack, or None when absent."""
        self._store.get_summary(run_id)
        return self._store.read_financial_analysis_pack(run_id)

    def read_chat_messages(self, run_id: str) -> list[dict]:
        """Persisted Q&A chat history for a run."""
        self._store.get_summary(run_id)
        return self._store.read_chat_messages(run_id)

    def append_chat_messages(self, run_id: str, messages: list[dict]) -> None:
        """Persist one or more Q&A chat messages for a run."""
        self._store.get_summary(run_id)
        self._store.append_chat_messages(run_id, messages)

    def report_package_path(self, run_id: str) -> Path:
        """Report-package location for the Report Q&A endpoint."""
        self._store.get_summary(run_id)
        return self._store.report_package_path(run_id)

    def set_archived(self, run_id: str, archived: bool) -> RunSummary:
        self._store.get_summary(run_id)
        return self._store.set_archived(run_id, archived)

    def delete_run(self, run_id: str) -> None:
        self._store.get_summary(run_id)
        self._store.delete_run(run_id)

    def read_report(self, run_id: str) -> str:
        return self._store.read_report(run_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(_STOP)
            for condition in self._conditions.values():
                condition.notify_all()
        self._worker.join(timeout=2)

    @property
    def worker_alive(self) -> bool:
        return self._worker.is_alive()

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                self._execute(str(item))
            finally:
                self._queue.task_done()

    def _execute(self, run_id: str) -> None:
        with self._lock:
            self._active_run_id = run_id
            request = self._requests[run_id]

        last_state: dict[str, Any] = {}
        try:
            self._persist_projection(
                run_id,
                project_state(last_state),
                status="running",
                report_available=False,
                message="正在进行：材料摄取",
            )
            run_directory = self._store.upload_directory(run_id).parent
            for update in self._runner.run(request, run_directory):
                last_state = update.state
                projection = project_state(last_state)
                if update.final_state is None:
                    self._persist_projection(
                        run_id,
                        projection,
                        status="running",
                        report_available=False,
                        message=self._progress_message(projection),
                    )
                    continue

                if projection.progress != 100:
                    raise RuntimeError("AI4Fin graph ended before the final stage completed")
                if update.report_markdown:
                    self._store.write_report(run_id, update.report_markdown)
                self._persist_projection(
                    run_id,
                    projection,
                    status="completed",
                    report_available=bool(update.report_markdown),
                    message="研究已完成",
                )
        except Exception as exc:
            logger.exception("AI4Fin web run failed: %s", run_id)
            message = self._safe_error_message(exc)
            self._persist_projection(
                run_id,
                project_state(last_state, error=message),
                status="failed",
                report_available=False,
                message=message,
            )
        finally:
            with self._lock:
                self._requests.pop(run_id, None)
                self._active_run_id = None

    def _persist_projection(
        self,
        run_id: str,
        projection: ProjectionResult,
        *,
        status: str,
        report_available: bool,
        message: str,
    ) -> None:
        for detail in projection.details.values():
            self._store.write_stage_detail(run_id, detail)
        previous = self._store.get_summary(run_id)
        summary = previous.model_copy(
            update={
                "status": status,
                "current_stage_id": projection.current_stage_id,
                "progress": projection.progress,
                "report_available": report_available,
                "stages": projection.stages,
            }
        )
        self._store.update_summary(summary)
        self._publish(run_id, message)

    def _publish(self, run_id: str, message: str) -> None:
        with self._lock:
            self._messages[run_id] = message
            self._versions[run_id] = self._versions.get(run_id, 0) + 1
            condition = self._conditions.get(run_id)
            if condition is not None:
                condition.notify_all()

    @staticmethod
    def _event_from_summary(summary: RunSummary, message: str) -> RunEvent:
        return RunEvent(
            run_id=summary.id,
            status=summary.status,
            current_stage_id=summary.current_stage_id,
            progress=summary.progress,
            report_available=summary.report_available,
            stages=summary.stages,
            message=message,
        )

    @staticmethod
    def _progress_message(projection: ProjectionResult) -> str:
        if projection.current_stage_id is None:
            return "正在整理最终结果"
        title = next(
            stage.title for stage in projection.stages if stage.id == projection.current_stage_id
        )
        return f"正在进行：{title}"

    @staticmethod
    def _safe_error_message(exc: Exception) -> str:
        compact = " ".join(str(exc).split()) or "研究任务运行失败"
        return compact[:240]

    @staticmethod
    def _attachment_names(uploads: Sequence[UploadPayload]) -> list[str]:
        names: list[str] = []
        for upload in uploads:
            name = Path(upload.filename).name
            if name in {"", ".", ".."}:
                raise ValueError("Upload filename is empty or unsafe")
            if name in names:
                raise FileExistsError(f"Duplicate upload filename: {name}")
            names.append(name)
        return names
