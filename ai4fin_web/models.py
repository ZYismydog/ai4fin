"""Pydantic contracts for the AI4Fin web API."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

RunStatus = Literal["queued", "running", "completed", "failed", "interrupted"]
StageStatus = Literal["pending", "running", "completed", "failed"]


class StageSummary(BaseModel):
    id: str
    title: str
    status: StageStatus


class RunSummary(BaseModel):
    id: str
    subject: str
    analysis_date: date
    query: str
    status: RunStatus
    current_stage_id: str | None
    progress: int = Field(ge=0, le=100)
    report_available: bool
    created_at: datetime
    stages: list[StageSummary]
    attachment_names: list[str]
    # Research configuration snapshot, mirroring the CLI selections so a
    # restored history run can replay the exact setup.
    config: dict = Field(default_factory=dict)
    # Archived runs are hidden from the default history but can be restored.
    archived: bool = False


class StageDetail(BaseModel):
    id: str
    title: str
    status: StageStatus
    available: bool
    summary: str = ""
    markdown: str = ""
    evidence: str = ""
    next_action: str = ""
    error: str = ""


class CreateRunInput(BaseModel):
    subject: str
    analysis_date: date
    query: str
    # Company display name when the subject was resolved from a free-form name.
    company_name: str | None = None
    # Research configuration (mirrors cli.main.get_user_selections). All
    # optional: omitted fields fall back to the server defaults.
    selected_analysts: list[str] = Field(default_factory=lambda: ["market", "news", "fundamentals"])
    research_depth: int | None = Field(default=None, ge=1, le=5)
    # Annual-report years (mirror cli/utils get_report_year + get_comparison_report_year).
    analysis_year: int | None = Field(default=None, ge=1900)
    comparison_year: int | None = Field(default=None, ge=1900)
    target_metrics: str = ""
    data_scope: str = ""
    task_boundary: str = ""
    # None (default) means "keep the server-side configuration" — the browser
    # never overrides the terminal's .env settings unless explicitly asked.
    language: str | None = None
    llm_provider: str | None = None
    backend_url: str | None = None
    quick_think_llm: str | None = None
    deep_think_llm: str | None = None
    google_thinking_level: str | None = None
    openai_reasoning_effort: str | None = None
    anthropic_effort: str | None = None


class UploadPayload(BaseModel):
    filename: str
    content: bytes


class QARequest(BaseModel):
    """One Report Q&A turn (mirrors cli/main.run_report_qa_session)."""

    question: str
    action: str = "answer_only"  # answer_only | appendix | update_report
    allow_external_research: bool = False


class RunEvent(BaseModel):
    run_id: str
    status: RunStatus
    current_stage_id: str | None
    progress: int = Field(ge=0, le=100)
    report_available: bool
    stages: list[StageSummary]
    message: str = ""
