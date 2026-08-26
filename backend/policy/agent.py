"""
AI Decision Agent — brief §8.

Receives structured input (customer context, ML predictions, expected value
per candidate, compliance state) and returns structured JSON only:
{"action": ..., "reason": ...}. `action` MUST come from the explicit
allowed-action list for this case — enforced here by validation, never
trusted from the model's raw output.

This is a decision-and-explanation layer, not the final authority: the hard
compliance gate (brief §9, build-order step 5 — not implemented yet) runs
AFTER this and can override/block whatever this returns.

Provider-agnostic: LLMProvider is an abstract interface; concrete
implementations exist for Gemini, Groq, and a RuleBasedFallbackProvider that
needs no API key. Provider selection is via the LLM_PROVIDER env var
(default: auto-detect from which API key, if any, is set; falls back to the
rule-based provider). The fallback is NEVER silently presented as an LLM
decision — its output is always tagged provider="rule_based_fallback" so the
audit trail can't confuse the two.

NOTE: GeminiProvider and GroqProvider are implemented against each API's
documented request/response contract as of this writing, but have not been
exercised against a live API key in this environment (none is configured
here). Verify against current docs before depending on them for a demo —
API surfaces, and especially the default model IDs below, do drift.
"""

from __future__ import annotations

import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from features import ALL_INTERVENTIONS  # noqa: E402

from ev import CandidateAction  # noqa: E402


@dataclass
class DecisionInput:
    transaction_id: str
    customer_context: dict
    transaction_context: dict
    compliance_state: dict
    candidates: list  # list[CandidateAction], already EV-ranked and eligibility-filtered

    def allowed_actions(self) -> list[str]:
        return [c.action for c in self.candidates]

    def to_prompt_payload(self) -> dict:
        """The exact structured payload shown to the LLM."""
        return {
            "transaction_id": self.transaction_id,
            "customer_context": self.customer_context,
            "transaction_context": self.transaction_context,
            "compliance_state": self.compliance_state,
            "candidate_actions_ev_ranked": [c.as_dict() for c in self.candidates],
            "allowed_actions": self.allowed_actions(),
        }


@dataclass
class AgentDecision:
    action: str
    reason: str
    provider: str
    raw_response: Optional[str] = None
    valid: bool = True
    validation_note: Optional[str] = None


SYSTEM_PROMPT = """You are a revenue-recovery decision agent. You will be given a customer's \
context, ML-predicted recovery probabilities and expected values for a fixed set of allowed \
recovery actions, and the customer's compliance state. Choose the single best action from \
`allowed_actions` only — you may never invent an action that is not in that list, even if you \
think it would work better. Prefer the action with the highest expected_value unless there is a \
clear, stated reason (visible in the given context) to deviate — if you deviate, your reason must \
name that evidence. Respond with JSON only, matching exactly this shape: \
{"action": "<one of allowed_actions>", "reason": "<one to two sentences, referencing specific \
numbers from the input>"}. No other text, no markdown fencing."""


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def decide(self, decision_input: DecisionInput) -> AgentDecision:
        ...


class RuleBasedFallbackProvider(LLMProvider):
    """
    Deterministic stand-in used when no LLM API key is configured. Picks the
    highest-EV eligible candidate (candidates arrive already EV-sorted) and
    produces a reason string grounded in the actual numbers. This is NOT an
    LLM call — never represent it as one; provider is always tagged
    "rule_based_fallback" so the audit trail can't confuse the two.
    """
    name = "rule_based_fallback"

    def decide(self, decision_input: DecisionInput) -> AgentDecision:
        top = decision_input.candidates[0]
        reason = (
            f"Highest expected value (Rs {top.expected_value:.2f}) among "
            f"{len(decision_input.candidates)} allowed candidate(s); predicted recovery "
            f"probability {top.predicted_probability:.1%} on Rs {top.recoverable_amount:.2f} at risk."
        )
        return AgentDecision(action=top.action, reason=reason, provider=self.name, raw_response=None)


def _extract_json_object(text: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM text response
    (handles markdown code fences some models wrap JSON output in)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response: {text!r}")
    return json.loads(text[start:end + 1])


def validate_decision(raw: dict, decision_input: DecisionInput, provider_name: str, raw_response: str) -> AgentDecision:
    """
    Never trust the model's chosen action at face value — the brief is
    explicit that `action` must come from the allowed list, never invented.
    An invalid/missing action is corrected to the top EV-ranked candidate,
    with valid=False so this is visible in the audit trail rather than
    silently patched over.
    """
    allowed = decision_input.allowed_actions()
    action = raw.get("action") if isinstance(raw, dict) else None
    reason = raw.get("reason", "").strip() if isinstance(raw, dict) and isinstance(raw.get("reason"), str) else ""

    if action in allowed and reason:
        return AgentDecision(action=action, reason=reason, provider=provider_name, raw_response=raw_response, valid=True)

    fallback_action = decision_input.candidates[0].action
    note = (
        f"Model returned invalid/incomplete decision (action={action!r}, allowed={allowed}); "
        f"corrected to top EV candidate {fallback_action!r}."
    )
    return AgentDecision(
        action=fallback_action,
        reason=reason or "(no reason returned)",
        provider=provider_name,
        raw_response=raw_response,
        valid=False,
        validation_note=note,
    )


class GeminiProvider(LLMProvider):
    """Google AI Studio Gemini API (free tier: Flash-Lite). Reads GEMINI_API_KEY.
    Model id via GEMINI_MODEL env var (default below) — check current Google AI
    Studio docs before relying on this id; it has not been verified live here."""
    name = "gemini"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

    def decide(self, decision_input: DecisionInput) -> AgentDecision:
        import requests

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": json.dumps(decision_input.to_prompt_payload())}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
        }
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        raw = _extract_json_object(text)
        return validate_decision(raw, decision_input, self.name, text)


class GroqProvider(LLMProvider):
    """Groq's OpenAI-compatible chat completions API (free tier). Reads
    GROQ_API_KEY. Model id via GROQ_MODEL env var (default below) — check
    current Groq docs before relying on this id; it has not been verified
    live here."""
    name = "groq"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model or os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not set")

    def decide(self, decision_input: DecisionInput) -> AgentDecision:
        import requests

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(decision_input.to_prompt_payload())},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        raw = _extract_json_object(text)
        return validate_decision(raw, decision_input, self.name, text)


def get_provider(name: Optional[str] = None) -> LLMProvider:
    """
    Provider selection, env-var driven so no code changes are needed to
    switch: LLM_PROVIDER=gemini|groq|rule_based_fallback. With no explicit
    name (and no LLM_PROVIDER env var), auto-detects from which API key is
    set (Gemini first, then Groq), falling back to the deterministic
    provider if neither is configured — which is the case in this
    environment right now, so every demo run below uses it.
    """
    name = name or os.environ.get("LLM_PROVIDER")
    if name == "gemini":
        return GeminiProvider()
    if name == "groq":
        return GroqProvider()
    if name == "rule_based_fallback":
        return RuleBasedFallbackProvider()

    if os.environ.get("GEMINI_API_KEY"):
        return GeminiProvider()
    if os.environ.get("GROQ_API_KEY"):
        return GroqProvider()
    return RuleBasedFallbackProvider()
