"""Adapter that executes the real AI4Fin graph for a browser run."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tradingagents.default_config import DEFAULT_CONFIG

from .models import CreateRunInput

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunnerUpdate:
    state: dict[str, Any]
    final_state: dict[str, Any] | None = None
    report_markdown: str = ""


def _default_graph_factory(**kwargs: Any) -> Any:
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    return TradingAgentsGraph(**kwargs)


def _default_material_preparer(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from tradingagents.dataflows.research_sources import prepare_research_materials

    return prepare_research_materials(*args, **kwargs)


def _default_annual_report_preparer(request: CreateRunInput, material_dir: Path) -> dict[str, Any] | None:
    """Fetch annual reports from CNINFO (巨潮资讯网) via Firecrawl when the user
    provided report years but no uploads — mirrors the terminal's smart-search
    material step. Returns None when the provider is unavailable."""
    from tradingagents.dataflows.annual_report_sources import (
        FirecrawlAnnualReportProvider,
        prepare_annual_report_materials,
        prepare_comparative_annual_report_materials,
    )

    provider = FirecrawlAnnualReportProvider.from_env()
    if not provider._is_configured():
        return None
    company = (request.company_name or request.subject or "").strip()
    if request.analysis_year and request.comparison_year:
        return prepare_comparative_annual_report_materials(
            company,
            analysis_year=request.analysis_year,
            comparison_year=request.comparison_year,
            cache_dir=material_dir,
            provider=provider,
        )
    year = request.analysis_year or request.comparison_year
    if not year:
        return None
    return prepare_annual_report_materials(
        company, year, cache_dir=material_dir, provider=provider
    )


def build_run_config(request: CreateRunInput) -> dict[str, Any]:
    """Assemble the graph config from a web run, mirroring the CLI's
    ``_build_run_config`` (cli/main.py) key-for-key.

    Rule: the server-side environment wins. Like the terminal, an explicit
    ``TRADINGAGENTS_*`` override is preserved even when the browser submits a
    selection — only unset keys are written from the request.
    """
    config = DEFAULT_CONFIG.copy()
    if request.research_depth is not None:
        # Research depth sets both round counts (cli/utils select_research_depth:
        # shallow=1 / medium=3 / deep=5). Explicit env overrides win (#977).
        if not os.environ.get("TRADINGAGENTS_MAX_DEBATE_ROUNDS"):
            config["max_debate_rounds"] = request.research_depth
        if not os.environ.get("TRADINGAGENTS_MAX_RISK_ROUNDS"):
            config["max_risk_discuss_rounds"] = request.research_depth
    # Every other overridable key follows the same rule: the server-side
    # TRADINGAGENTS_* environment wins, exactly as in the CLI where the
    # interactive step is skipped when the env var is already set.
    if request.llm_provider and not os.environ.get("TRADINGAGENTS_LLM_PROVIDER"):
        config["llm_provider"] = request.llm_provider
    if request.backend_url and not os.environ.get("TRADINGAGENTS_LLM_BACKEND_URL"):
        config["backend_url"] = request.backend_url
    if request.quick_think_llm and not os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM"):
        config["quick_think_llm"] = request.quick_think_llm
    if request.deep_think_llm and not os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM"):
        config["deep_think_llm"] = request.deep_think_llm
    if request.google_thinking_level and not os.environ.get("TRADINGAGENTS_GOOGLE_THINKING_LEVEL"):
        config["google_thinking_level"] = request.google_thinking_level
    if request.openai_reasoning_effort and not os.environ.get("TRADINGAGENTS_OPENAI_REASONING_EFFORT"):
        config["openai_reasoning_effort"] = request.openai_reasoning_effort
    if request.anthropic_effort and not os.environ.get("TRADINGAGENTS_ANTHROPIC_EFFORT"):
        config["anthropic_effort"] = request.anthropic_effort
    if request.language and not os.environ.get("TRADINGAGENTS_OUTPUT_LANGUAGE"):
        config["output_language"] = request.language
    return config


class Ai4FinRunner:
    """Stream graph states while keeping graph construction replaceable in tests."""

    def __init__(
        self,
        *,
        graph_factory: Callable[..., Any] = _default_graph_factory,
        material_preparer: Callable[..., Mapping[str, Any]] = _default_material_preparer,
        annual_report_preparer: Callable[..., Mapping[str, Any] | None] | None = _default_annual_report_preparer,
    ) -> None:
        self._graph_factory = graph_factory
        self._material_preparer = material_preparer
        self._annual_report_preparer = annual_report_preparer

    def run(self, request: CreateRunInput, run_directory: Path) -> Iterator[RunnerUpdate]:
        run_dir = run_directory.resolve()
        upload_dir = run_dir / "uploads"
        material_dir = run_dir / "materials"
        report_dir = run_dir / "reports"
        for directory in (upload_dir, material_dir, report_dir):
            directory.mkdir(parents=True, exist_ok=True)

        materials_path = ""
        material_context = ""
        if any(path.is_file() for path in upload_dir.iterdir()):
            prepared = self._material_preparer(upload_dir, cache_dir=material_dir)
            materials_path = str(upload_dir)
            material_context = str(prepared.get("context") or "")
        elif self._annual_report_preparer is not None and (
            request.analysis_year or request.comparison_year
        ):
            # No uploads but report years were provided: fetch the annual
            # report (CNINFO 巨潮资讯网 via Firecrawl) like the terminal does.
            try:
                annual = self._annual_report_preparer(request, material_dir)
                if annual and annual.get("materials_path"):
                    materials_path = str(annual["materials_path"])
                    material_context = str(annual.get("context") or "")
            except Exception as exc:
                logger.warning(
                    "Annual-report fetch failed for %r: %s", request.subject, exc
                )

        config = build_run_config(request)
        selected_analysts = tuple(request.selected_analysts or ("market", "news", "fundamentals"))
        graph = self._graph_factory(
            selected_analysts=selected_analysts,
            debug=False,
            config=config,
        )
        initial_state = graph.propagator.create_initial_state(
            company_name=request.subject,
            trade_date=request.analysis_date.isoformat(),
            analysis_task=request.query,
            target_metrics=request.target_metrics,
            data_scope=request.data_scope,
            task_boundary=request.task_boundary,
            language=request.language or config["output_language"],
            research_materials_path=materials_path,
            research_material_context=material_context,
            analysis_year=request.analysis_year,
            comparison_year=request.comparison_year,
        )
        graph_args = graph.propagator.get_graph_args()

        pending_state: dict[str, Any] | None = None
        for streamed_state in graph.graph.stream(initial_state, **graph_args):
            if pending_state is not None:
                yield RunnerUpdate(state=pending_state)
            pending_state = dict(streamed_state)

        if pending_state is None:
            raise RuntimeError("AI4Fin graph produced no state")

        report_path = Path(
            graph.save_reports(pending_state, request.subject, save_path=report_dir)
        )
        report_markdown = report_path.read_text(encoding="utf-8")
        yield RunnerUpdate(
            state=pending_state,
            final_state=pending_state,
            report_markdown=report_markdown,
        )
