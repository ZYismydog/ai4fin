"""Evidence-pack QA agent for the AI4Fin web workbench.

Answers questions about a finished run by grounding a dedicated chat model
(default ``deepseek-v4-flash``) in the run's financial analysis evidence pack
(reports/1_analysts/financial_analysis_pack.md). Pure small-talk / greetings
are rejected with an explicit "cannot answer" bubble instead of being fed to
the model.
"""

from __future__ import annotations

import os
import re

from langchain_core.messages import HumanMessage, SystemMessage

DEFAULT_QA_MODEL = "deepseek-v4-flash"
_QA_TIMEOUT_SECONDS = 45

# Tokens that indicate pure small talk when nothing substantive remains after
# stripping them from the question.
_SMALLTALK_TOKENS = (
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "hello",
    "hi",
    "hey",
    "在吗",
    "在不在",
    "早上好",
    "中午好",
    "下午好",
    "晚上好",
    "谢谢",
    "感谢",
    "thanks",
    "thank you",
    "are you there",
    "你是谁",
    "你叫什么",
    "再见",
    "拜拜",
    "bye",
    "goodbye",
    "随便聊聊",
    "没事了",
)
_STRIP_RE = re.compile(r"[!！?？。.~～\s,，、;；:：'\"“”‘’·\-]+")


def is_smalltalk(question: str) -> bool:
    """True when the question is pure greeting/small talk with no substance."""
    stripped = str(question or "").strip()
    if not stripped:
        return True
    rest = stripped.lower()
    for token in _SMALLTALK_TOKENS:
        rest = rest.replace(token.lower(), " ")
    rest = _STRIP_RE.sub("", rest)
    return not rest


def _unanswerable_reply(question: str, company_label: str) -> dict:
    return {
        "answer_mode": "unanswerable",
        "answer": (
            f"抱歉，我只能回答关于「{company_label or '该研究报告'}」的研报问题，"
            "无法处理这个问候/闲聊请求。你可以问我：公司营收、净利润、毛利率、"
            "资产负债、研发强度、现金流、应收账款周转等财务指标，或「年报里某业务线的情况」。"
        ),
        "question": question,
        "facts": [],
        "calculations": [],
        "assumptions": [],
        "citations": [],
        "confidence": "high",
        "next_action": "answer_only",
        "requires_confirmation": False,
        "report_id": "",
        "version": "evidence-qa",
    }


class EvidenceQAAgent:
    """Grounded Q&A over one run's financial analysis evidence pack.

    The primary model (default deepseek-v4-flash) is tried first; on failure
    or timeout it falls back to the main quick-think model so the interaction
    never hangs. Both model names are overridable via env.
    """

    def __init__(
        self,
        evidence_pack: str,
        company_label: str = "",
        *,
        model: str | None = None,
        fallback_models: list[str] | None = None,
    ) -> None:
        self.evidence_pack = str(evidence_pack or "")
        self.company_label = company_label or ""
        self.model = model or os.environ.get("AI4FIN_QA_MODEL") or DEFAULT_QA_MODEL
        configured_fallback = os.environ.get("AI4FIN_QA_FALLBACK_MODEL")
        if fallback_models:
            self.fallback_models = list(fallback_models)
        elif configured_fallback:
            self.fallback_models = [configured_fallback]
        else:
            self.fallback_models = [
                m
                for m in (
                    os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM"),
                    os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM"),
                )
                if m
            ]
        self.last_error: str = ""

    def ask(self, question: str, *, report_id: str = "") -> dict:
        question = str(question or "").strip()
        if not question:
            raise ValueError("question is required")
        if is_smalltalk(question):
            return _unanswerable_reply(question, self.company_label)
        try:
            answer = self._ask_llm(question)
        except Exception as exc:  # noqa: BLE001 - surface a friendly bubble
            return {
                "answer_mode": "qa_error",
                "answer": (
                    f"暂时无法回答这个问题（模型调用失败：{type(exc).__name__}）。"
                    "请稍后重试，或换一种问法。"
                ),
                "question": question,
                "facts": [],
                "calculations": [],
                "assumptions": [],
                "citations": [],
                "confidence": "low",
                "next_action": "answer_only",
                "requires_confirmation": False,
                "report_id": report_id,
                "version": "evidence-qa",
            }
        return {
            "answer_mode": "evidence_qa",
            "answer": answer,
            "question": question,
            "facts": [],
            "calculations": [],
            "assumptions": [],
            "citations": [],
            "confidence": "high",
            "next_action": "answer_only",
            "requires_confirmation": False,
            "report_id": report_id,
            "version": "evidence-qa",
        }

    def _ask_llm(self, question: str) -> str:
        system = (
            "你是「投研分析助手」的证据问答助手。用户只会在研究报告完成后向你提问。\n"
            "规则：\n"
            "1. 只能基于下方『财务分析证据包』回答；证据包没有的信息，明确回答"
            "『证据包中无此信息』，不要编造、不要推测。\n"
            "2. 涉及指标时引用证据包里的数字与单位；区分报告年度与对比年度。\n"
            "3. 回答保持简洁（一般 3-8 句），用中文。\n\n"
            f"研究标的：{self.company_label or '未知'}\n\n"
            "『财务分析证据包』开始\n"
            f"{self.evidence_pack}\n"
            "『财务分析证据包』结束"
        )
        messages = [SystemMessage(content=system), HumanMessage(content=question)]
        last_exc: Exception | None = None
        for model in [self.model, *self.fallback_models]:
            try:
                llm = self._build_llm(model)
                response = llm.invoke(messages)
                content = str(response.content).strip()
                if content:
                    return content
            except Exception as exc:  # noqa: BLE001 - try the next model
                self.last_error = f"{model}: {type(exc).__name__}: {exc}"
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("模型返回为空")

    def _build_llm(self, model: str):
        """Plain ChatOpenAI with a hard timeout so a flaky relay never hangs."""
        from langchain_openai import ChatOpenAI

        base_url = os.environ.get("TRADINGAGENTS_LLM_BACKEND_URL")
        api_key = (
            os.environ.get("OPENAI_COMPATIBLE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "sk-local"
        )
        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            request_timeout=_QA_TIMEOUT_SECONDS,
            max_retries=1,
        )
