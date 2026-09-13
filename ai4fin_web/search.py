"""Fuzzy subject search for the research composer.

Users often type company names (中文/English/partial) instead of tickers.
The search returns a short list of candidate subjects built from a built-in
alias table (exact/prefix/substring matching) and, when reachable, Yahoo
Finance suggestions. Yahoo failures (rate limits, offline) degrade silently to
the local table so the composer always has something to offer.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# symbol -> (display name, aliases...)
SUBJECT_CATALOG: dict[str, tuple[str, tuple[str, ...]]] = {
    "NVDA": ("NVIDIA Corporation（英伟达）", ("nvidia", "英伟达", "nvda")),
    "AAPL": ("Apple Inc.（苹果）", ("apple", "苹果", "aapl")),
    "MSFT": ("Microsoft Corporation（微软）", ("microsoft", "微软", "msft")),
    "GOOGL": ("Alphabet Inc.（谷歌）", ("google", "alphabet", "谷歌", "googl")),
    "AMZN": ("Amazon.com Inc.（亚马逊）", ("amazon", "亚马逊", "amzn")),
    "META": ("Meta Platforms（脸书）", ("meta", "facebook", "脸书", "meta")),
    "TSLA": ("Tesla Inc.（特斯拉）", ("tesla", "特斯拉", "tsla")),
    "AMD": ("Advanced Micro Devices（超威半导体）", ("amd", "超威", "amd")),
    "INTC": ("Intel Corporation（英特尔）", ("intel", "英特尔", "intc")),
    "AVGO": ("Broadcom Inc.（博通）", ("broadcom", "博通", "avgo")),
    "NFLX": ("Netflix Inc.（奈飞）", ("netflix", "奈飞", "nflx")),
    "ORCL": ("Oracle Corporation（甲骨文）", ("oracle", "甲骨文", "orcl")),
    "CRM": ("Salesforce Inc.", ("salesforce", "crm")),
    "ADBE": ("Adobe Inc.（奥多比）", ("adobe", "adbe")),
    "PLTR": ("Palantir Technologies", ("palantir", "pltr")),
    "BABA": ("Alibaba Group（阿里巴巴）", ("alibaba", "阿里巴巴", "baba")),
    "PDD": ("PDD Holdings（拼多多）", ("pdd", "拼多多")),
    "JD": ("JD.com（京东）", ("jd", "京东", "jd.com")),
    "NIO": ("NIO Inc.（蔚来）", ("nio", "蔚来")),
    "XPEV": ("XPeng Inc.（小鹏汽车）", ("xpeng", "小鹏", "xpev")),
    "LI": ("Li Auto（理想汽车）", ("li auto", "理想", "li")),
    "BIDU": ("Baidu Inc.（百度）", ("baidu", "百度", "bidu")),
    "TCEHY": ("Tencent Music (ADR)", ("tencent music", "tcehy")),
    "0700.HK": ("Tencent Holdings（腾讯控股）", ("tencent", "腾讯", "0700.hk")),
    "9988.HK": ("Alibaba (HK)（阿里巴巴）", ("alibaba hk", "9988.hk")),
    "TSM": ("Taiwan Semiconductor（台积电）", ("tsmc", "台积电", "tsm")),
    "SMCI": ("Super Micro Computer（超微电脑）", ("supermicro", "超微", "smci")),
    "MU": ("Micron Technology（美光）", ("micron", "美光", "mu")),
    "QCOM": ("Qualcomm Inc.（高通）", ("qualcomm", "高通", "qcom")),
    "TXN": ("Texas Instruments（德州仪器）", ("texas instruments", "ti", "txn")),
    "SPY": ("SPDR S&P 500 ETF（标普500）", ("spy", "标普500", "s&p 500")),
    "QQQ": ("Invesco QQQ（纳斯达克100）", ("qqq", "纳斯达克100", "nasdaq 100")),
    "BTC-USD": ("Bitcoin（比特币）", ("bitcoin", "btc", "比特币", "btc-usd")),
    "ETH-USD": ("Ethereum（以太坊）", ("ethereum", "eth", "以太坊", "eth-usd")),
}

_MAX_LOCAL_RESULTS = 8

# Eastmoney suggestion endpoint: free, keyless, Chinese-name friendly, covers
# A-shares / HK / US. Public token used by the web suggest widget.
_EASTMONEY_SUGGEST_URL = "https://searchapi.eastmoney.com/api/suggest/get"
_EASTMONEY_TOKEN = "D43BF722C8E33BDC906FB84D85E326E8"

# Eastmoney MktNum -> canonical Yahoo-style suffix.
_MKT_NUM_SUFFIX = {
    1: ".SS",      # Shanghai A-shares (yfinance uses .SS)
    2: ".SZ",      # Shenzhen A-shares
    116: ".HK",    # Hong Kong (5-digit code loses its leading zero)
}
# Eastmoney MktNum -> readable market label for the UI.
_MKT_NUM_LABEL = {
    1: "沪A",
    2: "深A",
    105: "美股",
    106: "美股",
    107: "美股",
    116: "港股",
}
# Security-type fragments that indicate non-quotable instruments (sectors,
# indices, bonds, futures…). Everything else — including future/unknown labels
# such as 沪A/科创板 — is surfaced, so Chinese A-share results never get lost.
_DROPPED_SECURITY_TYPES = (
    "板块", "概念", "指数", "行业", "地域", "债券", "可转债", "期货", "外汇",
    "板块指数", "指数基金", "基金(LOF)",
)


def _eastmoney_symbol(row: dict) -> str | None:
    """Compose a Yahoo-style symbol from an Eastmoney suggestion row."""
    code = str(row.get("Code") or "").strip()
    if not code:
        return None
    security_type = str(row.get("SecurityTypeName") or "")
    if any(marker in security_type for marker in _DROPPED_SECURITY_TYPES):
        return None
    if code.startswith("BK") or code.startswith("R_"):
        return None  # sector/concept boards
    # Prefer MarketType (stable: 1=沪, 2=深, 116=港股, 105-107=美股); fall back
    # to MktNum which Eastmoney sometimes returns as "0" for A-shares.
    try:
        mkt = int(row.get("MarketType") or row.get("MktNum") or 0)
    except (TypeError, ValueError):
        return None
    suffix = _MKT_NUM_SUFFIX.get(mkt)
    if suffix == ".HK":
        # 00700 -> 0700.HK: Yahoo HK symbols keep 4 digits (drop one leading zero).
        trimmed = code[1:] if len(code) == 5 and code.startswith("0") else code
        return f"{trimmed}.HK"
    if suffix:
        return f"{code}{suffix}"
    if mkt in (105, 106, 107):  # US exchanges keep the raw symbol
        return code
    return None


def search_eastmoney(query: str, limit: int = 6) -> list[dict]:
    """Chinese-name friendly suggestions from Eastmoney; empty on any failure."""
    import json
    import urllib.parse
    import urllib.request

    try:
        url = f"{_EASTMONEY_SUGGEST_URL}?{urllib.parse.urlencode({
            'input': query, 'type': '14', 'token': _EASTMONEY_TOKEN, 'count': str(limit)})}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"},
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = json.load(response)
        rows = (payload.get("QuotationCodeTable") or {}).get("Data") or []
    except Exception as exc:
        logger.debug("Eastmoney subject search unavailable for %r: %s", query, exc)
        return []

    results: list[dict] = []
    for row in rows:
        symbol = _eastmoney_symbol(row)
        if not symbol:
            continue
        try:
            mkt = int(row.get("MarketType") or row.get("MktNum") or 0)
        except (TypeError, ValueError):
            mkt = None
        results.append(
            {
                "symbol": symbol,
                "name": str(row.get("Name") or ""),
                "exchange": _MKT_NUM_LABEL.get(mkt) or str(row.get("MarketType") or ""),
                "source": "eastmoney",
            }
        )
        if len(results) >= limit:
            break
    return results


def _normalize(text: str) -> str:
    return "".join(text.split()).lower()


def _match_score(query: str, symbol: str, aliases: tuple[str, ...]) -> int | None:
    """Rank a catalog entry against a normalized query; None = no match."""
    norm = _normalize(query)
    if not norm:
        return None
    if _normalize(symbol) == norm:
        return 0  # exact ticker
    if any(_normalize(alias) == norm for alias in aliases):
        return 1  # exact alias
    if _normalize(symbol).startswith(norm):
        return 2  # ticker prefix
    if any(alias.startswith(norm) for alias in (_normalize(a) for a in aliases)):
        return 3  # alias prefix
    if norm in _normalize(symbol) or any(norm in _normalize(a) for a in aliases):
        return 4  # substring
    return None


def search_local(query: str, limit: int = _MAX_LOCAL_RESULTS) -> list[dict]:
    """Candidates from the built-in catalog, best match first."""
    scored = [
        (score, symbol, name)
        for symbol, (name, aliases) in SUBJECT_CATALOG.items()
        if (score := _match_score(query, symbol, aliases)) is not None
    ]
    scored.sort(key=lambda item: item[0])
    return [
        {"symbol": symbol, "name": name, "exchange": "", "source": "local"}
        for _, symbol, name in scored[:limit]
    ]


def search_yahoo(query: str, limit: int = 5) -> list[dict]:
    """Online suggestions from Yahoo Finance; empty on any failure."""
    try:
        import yfinance as yf

        search = yf.Search(query, max_results=limit)
        results: list[dict] = []
        for quote in list(search.quotes or []):
            symbol = quote.get("symbol")
            if not symbol:
                continue
            name = quote.get("shortname") or quote.get("longname") or ""
            results.append(
                {
                    "symbol": str(symbol),
                    "name": str(name),
                    "exchange": str(quote.get("exchDisp") or quote.get("exchange") or ""),
                    "source": "yahoo",
                }
            )
        return results
    except Exception as exc:  # rate limits, offline, malformed payload
        logger.debug("Yahoo subject search unavailable for %r: %s", query, exc)
        return []


def search_subjects(query: str, limit: int = _MAX_LOCAL_RESULTS) -> list[dict]:
    """Merge candidates: built-in catalog, then Eastmoney (Chinese-friendly,
    A/HK/US). Yahoo is consulted only when the fast sources found nothing, so
    typing feels instant instead of blocking on a slow Yahoo round-trip.
    Deduplicated by symbol; broken online sources never block local results."""
    query = query.strip()
    if not query:
        return []
    combined: list[dict] = []
    seen: set[str] = set()

    def _add(items: list[dict]) -> None:
        for item in items:
            if item["symbol"] not in seen:
                combined.append(item)
                seen.add(item["symbol"])

    try:
        _add(search_local(query, limit=limit))
        _add(search_eastmoney(query, limit=limit))
    except Exception:  # defensive
        pass
    if not combined:
        try:
            _add(search_yahoo(query, limit=limit))
        except Exception:
            pass
    return combined[:limit]
