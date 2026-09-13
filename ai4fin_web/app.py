"""FastAPI service exposing AI4Fin jobs and the browser workbench."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from cli.utils import is_valid_ticker_input, normalize_ticker_symbol
from tradingagents.dataflows.report_qa_service import handle_report_qa_request

from .jobs import JobManager
from .models import CreateRunInput, QARequest, RunSummary, StageDetail, UploadPayload
from .options import build_run_options
from .qa_agent import EvidenceQAAgent
from .runner import Ai4FinRunner
from .search import search_subjects
from .storage import ReportUnavailable, RunNotFound, RunStore
from .subject_resolution import SubjectResolutionError, resolve_subject

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".txt"}
VALID_ANALYST_KEYS = {"market", "news", "fundamentals", "social"}


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _validate_report_years(analysis_year: int | None, comparison_year: int | None) -> None:
    """Mirror the CLI's year rules: 1900..current year, comparison != analysis."""
    current_year = date.today().year
    for label, value in (("analysis_year", analysis_year), ("comparison_year", comparison_year)):
        if value is not None and not 1900 <= value <= current_year:
            raise HTTPException(
                status_code=422,
                detail=f"{label} 需在 1900-{current_year} 之间",
            )
    if (
        analysis_year is not None
        and comparison_year is not None
        and comparison_year == analysis_year
    ):
        raise HTTPException(status_code=422, detail="对比年份不能与研究年份相同")


def _available_provider_keys() -> set[str]:
    """Concrete provider keys the server is configured to use."""
    keys: set[str] = set()
    for provider in build_run_options()["providers"]:
        if provider["available"]:
            if provider["regions"]:
                keys.update(r["key"] for r in provider["regions"] if r["available"])
            else:
                keys.add(provider["key"])
    return keys


class BrowserStaticFiles(StaticFiles):
    """Serve ES modules with a stable MIME type across operating systems."""

    async def get_response(self, path: str, scope: dict[str, Any]) -> Any:
        response = await super().get_response(path, scope)
        if Path(path).suffix.lower() in {".js", ".mjs"} and response.status_code == 200:
            response.headers["content-type"] = "text/javascript; charset=utf-8"
        return response


def _parse_analysts(raw: str | None) -> list[str]:
    """Parse the comma-separated analyst list, keeping mandatory ordering."""
    if raw is None or not raw.strip():
        return ["market", "news", "fundamentals"]
    parsed = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [key for key in parsed if key not in VALID_ANALYST_KEYS]
    if unknown:
        raise HTTPException(status_code=422, detail=f"未知分析师：{', '.join(unknown)}")
    if "market" not in parsed or "news" not in parsed or "fundamentals" not in parsed:
        raise HTTPException(status_code=422, detail="market/news/fundamentals 分析师为必选")
    return parsed


def _validate_provider(provider: str | None) -> str | None:
    """Ensure a chosen provider is actually configured server-side."""
    if provider is None or not provider.strip():
        return None
    provider_key = provider.strip().lower()
    if provider_key not in _available_provider_keys():
        raise HTTPException(
            status_code=422,
            detail=f"该模型提供商未在服务端配置（{provider_key}），请先在 .env 中配置 API key",
        )
    return provider_key


def _validate_model_ids(
    provider: str | None, quick_think_llm: str | None, deep_think_llm: str | None
) -> None:
    """Reject empty model IDs for providers that require a concrete model."""
    if provider is None:
        return
    for label, value in (("quick_think_llm", quick_think_llm), ("deep_think_llm", deep_think_llm)):
        if value is not None and not value.strip():
            raise HTTPException(status_code=422, detail=f"{label} 不能为空")


def create_app(
    *,
    manager: Any | None = None,
    web_directory: Path | None = None,
    data_root: Path | None = None,
) -> FastAPI:
    project_root = Path(__file__).resolve().parent.parent
    static_root = (web_directory or project_root / "web").resolve()
    owns_manager = manager is None
    job_manager = manager or JobManager(
        store=RunStore(data_root or project_root / "data" / "web_runs"),
        runner=Ai4FinRunner(),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if owns_manager:
            job_manager.close()

    app = FastAPI(title="AI4Fin Web", lifespan=lifespan)
    app.state.job_manager = job_manager

    @app.exception_handler(RunNotFound)
    async def run_not_found_handler(_: Request, exc: RunNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ReportUnavailable)
    async def report_unavailable_handler(
        _: Request, exc: ReportUnavailable
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/options")
    def options() -> dict:
        """Server-side run options: providers with configured keys, languages,
        research depths, model catalog, and defaults — mirrors the CLI flow."""
        return build_run_options()

    @app.get("/api/search")
    def search_subjects_endpoint(q: str = "") -> list[dict]:
        """Fuzzy subject search for the composer's autocomplete."""
        if not q.strip():
            return []
        return search_subjects(q.strip(), limit=8)

    @app.post("/api/runs", response_model=RunSummary, status_code=202)
    async def create_run(
        subject: Annotated[str, Form()],
        analysis_date: Annotated[date, Form()],
        query: Annotated[str, Form()],
        files: Annotated[list[UploadFile] | None, File()] = None,
        selected_analysts: Annotated[str | None, Form()] = None,
        research_depth: Annotated[int | None, Form()] = None,
        target_metrics: Annotated[str | None, Form()] = None,
        data_scope: Annotated[str | None, Form()] = None,
        task_boundary: Annotated[str | None, Form()] = None,
        language: Annotated[str | None, Form()] = None,
        llm_provider: Annotated[str | None, Form()] = None,
        backend_url: Annotated[str | None, Form()] = None,
        quick_think_llm: Annotated[str | None, Form()] = None,
        deep_think_llm: Annotated[str | None, Form()] = None,
        google_thinking_level: Annotated[str | None, Form()] = None,
        openai_reasoning_effort: Annotated[str | None, Form()] = None,
        anthropic_effort: Annotated[str | None, Form()] = None,
        analysis_year: Annotated[int | None, Form()] = None,
        comparison_year: Annotated[int | None, Form()] = None,
    ) -> RunSummary:
        normalized_subject = subject.strip()
        normalized_query = query.strip()
        if not normalized_subject:
            raise HTTPException(status_code=422, detail="请输入有效的研究对象或数据代码")
        if not normalized_query:
            raise HTTPException(status_code=422, detail="请输入研究问题")

        # A free-form company name (e.g. 寒武纪) is resolved to a canonical
        # symbol — local catalog, Eastmoney, then Firecrawl discovery — and a
        # failure is reported back as a friendly error.
        if not is_valid_ticker_input(normalized_subject) or _contains_cjk(normalized_subject):
            try:
                resolved = resolve_subject(normalized_subject)
            except SubjectResolutionError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            normalized_subject = resolved["symbol"]
            company_name = resolved.get("company_name") or normalized_subject
        else:
            company_name = normalized_subject

        _validate_report_years(analysis_year, comparison_year)
        analysts = _parse_analysts(selected_analysts)
        provider = _validate_provider(llm_provider)
        _validate_model_ids(provider, quick_think_llm, deep_think_llm)

        uploads: list[UploadPayload] = []
        for upload in files or []:
            filename = Path(upload.filename or "").name
            extension = Path(filename).suffix.lower()
            if not filename or extension not in SUPPORTED_UPLOAD_EXTENSIONS:
                await upload.close()
                raise HTTPException(status_code=415, detail=f"不支持的文件类型：{filename}")
            content = await upload.read(MAX_UPLOAD_BYTES + 1)
            await upload.close()
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"文件超过 20 MiB：{filename}")
            uploads.append(UploadPayload(filename=filename, content=content))

        request_model = CreateRunInput(
            subject=normalize_ticker_symbol(normalized_subject),
            analysis_date=analysis_date,
            query=normalized_query,
            company_name=company_name if company_name != normalized_subject else None,
            selected_analysts=analysts,
            research_depth=research_depth,
            analysis_year=analysis_year,
            comparison_year=comparison_year,
            target_metrics=(target_metrics or "").strip(),
            data_scope=(data_scope or "").strip(),
            task_boundary=(task_boundary or "").strip(),
            language=(language or "").strip() or None,
            llm_provider=provider,
            backend_url=(backend_url or "").strip() or None,
            quick_think_llm=(quick_think_llm or "").strip() or None,
            deep_think_llm=(deep_think_llm or "").strip() or None,
            google_thinking_level=(google_thinking_level or "").strip() or None,
            openai_reasoning_effort=(openai_reasoning_effort or "").strip() or None,
            anthropic_effort=(anthropic_effort or "").strip() or None,
        )
        try:
            return job_manager.submit(request_model, uploads)
        except (FileExistsError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/runs", response_model=list[RunSummary])
    def list_runs(include_archived: bool = False) -> list[RunSummary]:
        return job_manager.list(include_archived=include_archived)

    @app.post("/api/runs/{run_id}/archive", response_model=RunSummary)
    def set_run_archived(run_id: str, archived: bool = True) -> RunSummary:
        """Archive (or restore) a run in the history list."""
        try:
            return job_manager.set_archived(run_id, archived)
        except RunNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/runs/{run_id}", status_code=204)
    def delete_run(run_id: str) -> None:
        """Permanently delete a run and all of its artifacts."""
        try:
            job_manager.delete_run(run_id)
        except RunNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}", response_model=RunSummary)
    def get_run(run_id: str) -> RunSummary:
        return job_manager.get(run_id)

    @app.get("/api/runs/{run_id}/events")
    def stream_events(run_id: str) -> StreamingResponse:
        job_manager.get(run_id)

        def frames() -> Iterator[str]:
            for event in job_manager.subscribe(run_id):
                yield f"data: {event.model_dump_json()}\n\n"

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/runs/{run_id}/stages/{stage_id}", response_model=StageDetail)
    def get_stage_detail(run_id: str, stage_id: str) -> StageDetail:
        try:
            detail = job_manager.read_stage(run_id, stage_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # When the run finished, the report tree holds each agent's complete
        # output — surface it as the stage's full body in the trace drawer.
        artifact = job_manager.read_stage_artifact(run_id, stage_id)
        if artifact:
            detail = detail.model_copy(
                update={"markdown": artifact, "available": True}
            )
        return detail

    @app.get("/api/runs/{run_id}/report")
    def get_report(run_id: str) -> PlainTextResponse:
        # The user-facing report is the final review section of the report
        # tree (5_final/decision.md); older runs fall back to report.md.
        final_review = job_manager.read_final_review(run_id)
        if final_review:
            return PlainTextResponse(
                final_review,
                media_type="text/markdown; charset=utf-8",
            )
        return PlainTextResponse(
            job_manager.read_report(run_id),
            media_type="text/markdown; charset=utf-8",
        )

    @app.post("/api/runs/{run_id}/qa", response_model=dict)
    def ask_question(run_id: str, payload: QARequest) -> dict:
        """One Report Q&A turn grounded in the run's evidence pack.

        A dedicated QA agent (default model deepseek-v4-flash) answers every
        question using the financial analysis pack as context; pure greetings
        / small talk get an explicit "cannot answer" bubble. Without a pack,
        falls back to the LLM package session (cli/main.run_report_qa_session
        semantics: ``:appendix`` maps to action ``appendix``, ``:update`` to
        ``update_report``)."""
        pack = job_manager.read_financial_analysis_pack(run_id)
        if pack is not None:
            summary = job_manager.get(run_id)
            agent = EvidenceQAAgent(pack, company_label=summary.subject)
            result = agent.ask(payload.question, report_id=run_id)
            job_manager.append_chat_messages(
                run_id,
                [
                    {"role": "user", "content": payload.question},
                    {
                        "role": "assistant",
                        "content": result.get("answer", ""),
                        "mode": result.get("answer_mode", "evidence_qa"),
                    },
                ],
            )
            return result
        package_path = job_manager.report_package_path(run_id)
        try:
            result = handle_report_qa_request(
                {
                    "package_path": str(package_path),
                    "question": payload.question,
                    "action": payload.action,
                    "allow_external_research": payload.allow_external_research,
                }
            )
            job_manager.append_chat_messages(
                run_id,
                [
                    {"role": "user", "content": payload.question},
                    {
                        "role": "assistant",
                        "content": result.get("answer", ""),
                        "mode": result.get("answer_mode", "answer_only"),
                    },
                ],
            )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/chat", response_model=list[dict])
    def chat_history(run_id: str) -> list[dict]:
        """Persisted Q&A chat history for a run, oldest first."""
        return job_manager.read_chat_messages(run_id)

    if not static_root.is_dir():
        raise RuntimeError(f"Web asset directory not found: {static_root}")
    app.mount("/", BrowserStaticFiles(directory=static_root, html=True), name="web")
    return app
