"""Resolve a free-form research subject (company name) to a canonical symbol.

Pipeline used when the composer's autocomplete was skipped and the user sends
a name directly (e.g. ``寒武纪``): try the local catalog, then Eastmoney, then
Firecrawl company discovery (feeding the discovered company name back into
Eastmoney for its ticker). Failure raises a user-facing error so the web layer
can reply "没有找到相关公司" instead of a 500.
"""

from __future__ import annotations

import logging

from cli.utils import is_valid_ticker_input

from .search import search_eastmoney, search_local

logger = logging.getLogger(__name__)


class SubjectResolutionError(ValueError):
    """Raised when a free-form subject cannot be mapped to a symbol."""


def _pick_symbol(items: list[dict]) -> dict | None:
    """First non-empty candidate (sources are pre-ranked)."""
    for item in items:
        if item.get("symbol"):
            return item
    return None


def _eastmoney_for(company_name: str) -> dict | None:
    try:
        return _pick_symbol(search_eastmoney(company_name, limit=6))
    except Exception:
        return None


def _firecrawl_company_name(query: str) -> str | None:
    """Discover the company's full name via Firecrawl; None when unavailable."""
    try:
        from tradingagents.dataflows.annual_report_sources import (
            FirecrawlAnnualReportProvider,
        )

        provider = FirecrawlAnnualReportProvider.from_env()
        if not provider.base_url and not provider.api_key:
            return None
        candidates = provider.discover_company_candidates(query, limit=4)
    except Exception as exc:
        logger.debug("Firecrawl company discovery failed for %r: %s", query, exc)
        return None
    if not candidates:
        return None
    for candidate in candidates:
        label = (candidate.label or "").strip()
        title = (candidate.title or "").strip()
        for name in (label, title):
            if name and any(ch.isalpha() for ch in name):
                return name
    return None


_ALLOWED_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-=^ ")


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _plausible_subject(raw: str) -> bool:
    """A subject is worth resolving when it is a plausible company name/ticker:
    CJK text, or alphanumerics with Yahoo-style separators. Garbage like
    ``NVDA/$`` is rejected immediately without network lookups."""
    if _contains_cjk(raw):
        return True
    return all(char in _ALLOWED_NAME_CHARS for char in raw) and any(
        char.isalnum() for char in raw
    )


def resolve_subject(subject: str) -> dict:
    """Map user input to ``{"symbol": ..., "company_name": ...}`` or raise.

    - A valid ticker passes through unchanged.
    - Otherwise: local catalog -> Eastmoney -> Firecrawl discovery (+Eastmoney).
    - Obvious garbage fails fast without any network lookup.
    """
    raw = subject.strip()
    if not raw:
        raise SubjectResolutionError("请输入研究对象或股票代码")

    if not _plausible_subject(raw):
        raise SubjectResolutionError(f"无法识别的研究对象：{raw}")

    if is_valid_ticker_input(raw) and not _contains_cjk(raw):
        return {"symbol": raw, "company_name": raw}

    # 1. Built-in catalog (covers common US/HK/CN names + ticker aliases).
    local = search_local(raw, limit=4)
    hit = _pick_symbol(local)
    if hit:
        return {"symbol": hit["symbol"], "company_name": hit["name"]}

    # 2. Eastmoney suggestion (Chinese-friendly, A/HK/US).
    eastmoney = _eastmoney_for(raw)
    if eastmoney:
        return {"symbol": eastmoney["symbol"], "company_name": eastmoney["name"]}

    # 3. Firecrawl company discovery, then Eastmoney for the ticker.
    company_name = _firecrawl_company_name(raw)
    if company_name:
        logger.info("Firecrawl resolved %r -> %s", raw, company_name)
        via_name = _eastmoney_for(company_name)
        if via_name:
            return {"symbol": via_name["symbol"], "company_name": company_name}

    raise SubjectResolutionError(f"没有找到相关公司：{raw}")


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)
