"""Project cumulative AI4Fin graph state into the nine-stage web workflow."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .models import StageDetail, StageSummary


@dataclass(frozen=True)
class StageSpec:
    id: str
    title: str
    fields: tuple[str, ...]
    any_field: bool = False


@dataclass(frozen=True)
class ProjectionResult:
    stages: list[StageSummary]
    details: dict[str, StageDetail]
    current_stage_id: str | None
    progress: int


STAGES = (
    StageSpec("ingestion", "材料摄取", ("document_ingestion_report",)),
    StageSpec("standardization", "任务标准化", ("standardized_task",)),
    StageSpec("decomposition", "问题拆解", ("question_decomposition",)),
    StageSpec("keywords", "关键词规划", ("keyword_plan",)),
    StageSpec(
        "evidence",
        "证据研究",
        (
            "market_report",
            "sentiment_report",
            "news_report",
            "fundamentals_report",
            "financial_analysis_pack",
        ),
        any_field=True,
    ),
    StageSpec(
        "debate",
        "研究辩论",
        ("investment_debate_state", "investment_plan"),
        any_field=True,
    ),
    StageSpec("writing", "报告撰写", ("trader_investment_plan",)),
    StageSpec("qa", "质量复核", ("risk_debate_state",)),
    StageSpec("final", "最终审阅", ("final_trade_decision",)),
)

_FIELD_TITLES = {
    "document_ingestion_report": "材料摄取报告",
    "standardized_task": "标准化任务",
    "question_decomposition": "问题拆解",
    "keyword_plan": "关键词规划",
    "market_report": "市场分析",
    "sentiment_report": "情绪分析",
    "news_report": "新闻分析",
    "fundamentals_report": "基本面分析",
    "financial_analysis_pack": "财务分析包",
    "investment_debate_state": "研究辩论记录",
    "investment_plan": "研究结论",
    "trader_investment_plan": "报告草稿",
    "risk_debate_state": "质量复核记录",
    "final_trade_decision": "最终报告",
}


def project_state(
    state: Mapping[str, Any],
    *,
    error: str = "",
) -> ProjectionResult:
    """Return summary-only stage status and a separate detail artifact map."""

    stages: list[StageSummary] = []
    details: dict[str, StageDetail] = {}
    current_stage_id: str | None = None
    sequence_blocked = False
    completed_count = 0

    for spec in STAGES:
        has_output = _stage_has_output(spec, state)
        if not sequence_blocked and has_output:
            status = "completed"
            completed_count += 1
        elif not sequence_blocked:
            status = "failed" if error else "running"
            current_stage_id = spec.id
            sequence_blocked = True
        else:
            status = "pending"

        stages.append(StageSummary(id=spec.id, title=spec.title, status=status))
        details[spec.id] = _build_detail(
            spec,
            state,
            status=status,
            error=error if status == "failed" else "",
        )

    progress = 100 if completed_count == len(STAGES) else int(completed_count * 100 / len(STAGES))
    return ProjectionResult(
        stages=stages,
        details=details,
        current_stage_id=current_stage_id,
        progress=progress,
    )


def _stage_has_output(spec: StageSpec, state: Mapping[str, Any]) -> bool:
    values = [_has_content(state.get(field)) for field in spec.fields]
    return any(values) if spec.any_field else all(values)


def _has_content(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_has_content(item) for key, item in value.items() if key != "count")
    if isinstance(value, (list, tuple, set)):
        return any(_has_content(item) for item in value)
    if isinstance(value, (int, float)):
        return value != 0
    return True


def _build_detail(
    spec: StageSpec,
    state: Mapping[str, Any],
    *,
    status: str,
    error: str,
) -> StageDetail:
    sections: list[str] = []
    evidence_fields: list[str] = []
    first_body = ""
    for field in spec.fields:
        value = state.get(field)
        if not _has_content(value):
            continue
        body = _render_value(value)
        if not first_body:
            first_body = body
        title = _FIELD_TITLES.get(field, field)
        sections.append(f"## {title}\n\n{body}")
        evidence_fields.append(title)

    available = status == "completed" or (status == "failed" and bool(error))
    if status == "completed":
        next_action = "流程完成" if spec.id == "final" else "进入下一环节"
    elif status == "failed":
        next_action = "检查配置或输入后重新运行"
    else:
        next_action = "等待该环节完成"

    return StageDetail(
        id=spec.id,
        title=spec.title,
        status=status,
        available=available,
        summary=_summarize(first_body),
        markdown="\n\n".join(sections),
        evidence="、".join(evidence_fields),
        next_action=next_action,
        error=error,
    )


def _render_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _summarize(body: str, limit: int = 180) -> str:
    compact = " ".join(body.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
