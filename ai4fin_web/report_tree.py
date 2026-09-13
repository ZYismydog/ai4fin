"""Map the nine web stages to the terminal's report tree.

``write_report_tree`` (tradingagents/reporting.py) saves each agent's full
output under ``reports/`` as ``0_preanalysis`` … ``5_final`` directories. The
browser's trace drawer shows these complete outputs per stage, while the main
chat view presents only the final review (``5_final/decision.md``).
"""

from __future__ import annotations

from pathlib import Path

# stage_id -> relative report-tree files whose contents make up that stage's
# complete agent output. Files that were not produced (e.g. sentiment analyst
# when not selected) are skipped automatically by _read_many.
STAGE_REPORT_FILES: dict[str, tuple[str, ...]] = {
    "ingestion": ("0_preanalysis/document_ingestion.md",),
    "standardization": ("0_preanalysis/task_standardizer.md",),
    "decomposition": ("0_preanalysis/question_decomposer.md",),
    "keywords": ("0_preanalysis/keyword_planner.md",),
    "evidence": (
        "1_analysts/market.md",
        "1_analysts/sentiment.md",
        "1_analysts/news.md",
        "1_analysts/fundamentals.md",
        "1_analysts/financial_analysis_pack.md",
    ),
    "debate": (
        "2_research/bull.md",
        "2_research/bear.md",
        "2_research/manager.md",
    ),
    "writing": ("3_draft_report/trader.md",),
    "qa": (
        "4_review/aggressive.md",
        "4_review/conservative.md",
        "4_review/neutral.md",
    ),
    "final": ("5_final/decision.md",),
}

# Headers prepended to each report-tree file inside a merged stage body.
_FILE_TITLES: dict[str, str] = {
    "1_analysts/market.md": "Market Conditions Analyst",
    "1_analysts/sentiment.md": "Sentiment Analyst",
    "1_analysts/news.md": "Compliance Standards Analyst",
    "1_analysts/fundamentals.md": "Research Materials Analyst",
    "1_analysts/financial_analysis_pack.md": "Annual Report Financial Analysis Pack",
    "2_research/bull.md": "Bull Researcher",
    "2_research/bear.md": "Bear Researcher",
    "2_research/manager.md": "Research Manager",
    "4_review/aggressive.md": "Aggressive Analyst",
    "4_review/conservative.md": "Conservative Analyst",
    "4_review/neutral.md": "Neutral Analyst",
    "5_final/decision.md": "Final Research Manager",
}

FINAL_REVIEW_FILE = "5_final/decision.md"


def _read_many(reports_dir: Path, files: tuple[str, ...]) -> str:
    """Concatenate existing report-tree files with readable section headers."""
    parts: list[str] = []
    for relative in files:
        path = reports_dir / relative
        if not path.is_file():
            continue
        title = _FILE_TITLES.get(relative, path.stem)
        parts.append(f"### {title}\n\n{path.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(parts)


def read_stage_report(reports_dir: Path, stage_id: str) -> str | None:
    """Full agent output for one web stage, or None when the tree lacks it."""
    files = STAGE_REPORT_FILES.get(stage_id)
    if not files:
        return None
    body = _read_many(reports_dir, files)
    return body or None


def read_final_review(reports_dir: Path) -> str | None:
    """The final review section shown as the user-facing report."""
    path = reports_dir / FINAL_REVIEW_FILE
    if not path.is_file():
        return None
    body = path.read_text(encoding="utf-8").strip()
    return body or None


def has_report_tree(reports_dir: Path) -> bool:
    """Whether a completed report tree exists under the run's reports dir."""
    return reports_dir.is_dir() and any(
        (reports_dir / file).is_file()
        for files in STAGE_REPORT_FILES.values()
        for file in files
    )
