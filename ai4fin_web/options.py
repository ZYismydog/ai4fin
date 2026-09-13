"""Server-side run options for the AI4Fin workbench.

Mirrors the interactive selections offered by the ``ai4fin`` CLI
(``cli/main.get_user_selections``) so the browser form can reproduce the
terminal flow. Provider availability is derived from the process environment:
a provider is offered only when its API-key env var is set (or the provider
needs no key), so secrets never travel through the browser.
"""

from __future__ import annotations

import os

from cli.utils import _llm_provider_table
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS

# Region variants: the CLI asks for a region after picking these providers;
# the browser form offers the same choice, resolving to the concrete key.
PROVIDER_REGIONS: dict[str, list[tuple[str, str]]] = {
    "qwen": [("International (dashscope-intl)", "qwen"), ("China (dashscope)", "qwen-cn")],
    "glm": [("Z.AI (international)", "glm"), ("BigModel (China)", "glm-cn")],
    "minimax": [("Global (api.minimax.io)", "minimax"), ("China (api.minimaxi.com)", "minimax-cn")],
}

# Language choices mirror cli.utils.ask_output_language; "custom" is resolved
# by the caller into a free-form language name.
LANGUAGES: list[tuple[str, str]] = [
    ("English (default)", "English"),
    ("Chinese (中文)", "Chinese"),
    ("Japanese (日本語)", "Japanese"),
    ("Korean (한국어)", "Korean"),
    ("Hindi (हिन्दी)", "Hindi"),
    ("Spanish (Español)", "Spanish"),
    ("Portuguese (Português)", "Portuguese"),
    ("French (Français)", "French"),
    ("German (Deutsch)", "German"),
    ("Arabic (العربية)", "Arabic"),
    ("Russian (Русский)", "Russian"),
    ("Custom language", "custom"),
]

# Research depth mirrors cli.utils.select_research_depth: the value doubles as
# both debate-round and risk-discussion-round count.
RESEARCH_DEPTHS: list[dict[str, int | str]] = [
    {"label": "Shallow - Quick research, few debate and strategy discussion rounds", "value": 1},
    {"label": "Medium - Middle ground, moderate debate rounds and strategy discussion", "value": 3},
    {"label": "Deep - Comprehensive research, in depth debate and strategy discussion", "value": 5},
]

# Terminal defaults for the free-form research fields (cli.utils helpers).
DEFAULT_ANALYSIS_TASK = "分析该研究对象的核心增长逻辑是否有充分证据支撑"
DEFAULT_TARGET_METRICS = "营收增长、毛利率、现金流、客户集中度、行业景气度"
DEFAULT_DATA_SCOPE = "公开财报、新闻、宏观数据、市场情绪"

# Provider-specific thinking configuration offered after model selection,
# mirroring cli/main.py:1124-1150.
PROVIDER_THINKING_OPTIONS: dict[str, list[dict[str, str]]] = {
    "google": [
        {"label": "High (thinking enabled)", "value": "high"},
        {"label": "Minimal (faster, lower latency)", "value": "minimal"},
    ],
    "openai": [
        {"label": "Medium (default)", "value": "medium"},
        {"label": "High (more thorough)", "value": "high"},
        {"label": "Low (faster)", "value": "low"},
    ],
    "anthropic": [
        {"label": "High (recommended)", "value": "high"},
        {"label": "Medium (balanced)", "value": "medium"},
        {"label": "Low (faster, cheaper)", "value": "low"},
    ],
}

# Which analyst keys the CLI keeps mandatory, in display order.
MANDATORY_ANALYSTS: tuple[str, ...] = ("market", "news", "fundamentals")
OPTIONAL_ANALYSTS: tuple[tuple[str, str], ...] = (
    ("social", "Sentiment Analyst (Xiaohongshu/Douyin/Zhihu evidence)"),
)


def _has_key(provider_key: str) -> bool:
    """Whether the provider's API key env var is set in this process."""
    env_var = get_api_key_env(provider_key)
    if env_var is None:
        # Local / credential-chain providers (ollama, bedrock, key-optional
        # openai-compatible relays) are offered without a key check.
        return True
    return bool(os.environ.get(env_var, "").strip())


def available_providers() -> list[dict]:
    """List every CLI provider with availability and region info.

    A provider is available when its key is configured (or no key is needed).
    Region providers (qwen/glm/minimax) are reported as a single entry whose
    ``regions`` list resolves to concrete provider keys.
    """
    providers: list[dict] = []
    for label, provider_key, default_url in _llm_provider_table():
        if provider_key in PROVIDER_REGIONS:
            regions = []
            for region_label, region_key in PROVIDER_REGIONS[provider_key]:
                regions.append(
                    {
                        "key": region_key,
                        "label": region_label,
                        "available": _has_key(region_key),
                    }
                )
            providers.append(
                {
                    "key": provider_key,
                    "label": label,
                    "available": any(region["available"] for region in regions),
                    "default_url": default_url,
                    "regions": regions,
                    "requires_key": any(
                        get_api_key_env(region["key"]) is not None
                        for region in regions
                    ),
                }
            )
        else:
            providers.append(
                {
                    "key": provider_key,
                    "label": label,
                    "available": _has_key(provider_key),
                    "default_url": default_url,
                    "regions": [],
                    "requires_key": get_api_key_env(provider_key) is not None,
                }
            )
    return providers


def model_catalog() -> dict[str, dict[str, list[dict[str, str]]]]:
    """Serialize the shared model catalog for the browser dropdowns."""
    return {
        provider: {
            mode: [{"label": label, "value": value} for label, value in options]
            for mode, options in mode_options.items()
        }
        for provider, mode_options in MODEL_OPTIONS.items()
    }


def build_run_options() -> dict:
    """Assemble the full options payload served by ``GET /api/options``."""
    return {
        "providers": available_providers(),
        "languages": [{"label": label, "value": value} for label, value in LANGUAGES],
        "depths": RESEARCH_DEPTHS,
        "mandatory_analysts": list(MANDATORY_ANALYSTS),
        "optional_analysts": [
            {"key": key, "label": label} for key, label in OPTIONAL_ANALYSTS
        ],
        "models": model_catalog(),
        "thinking": PROVIDER_THINKING_OPTIONS,
        "defaults": {
            "analysis_task": DEFAULT_ANALYSIS_TASK,
            "target_metrics": DEFAULT_TARGET_METRICS,
            "data_scope": DEFAULT_DATA_SCOPE,
            "language": "English",
            "depth": 1,
            "provider": "openai",
        },
    }
