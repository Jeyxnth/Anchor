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

NOTE: GeminiProvider has been exercised against a live GEMINI_API_KEY
(2026-08-26) and confirmed working end-to-end (request shape, JSON response
parsing, validate_decision()) on gemini-2.5-flash-lite — the original default
(gemini-2.0-flash-lite) 404'd, model retired/renamed since this was first
written; current model IDs were pulled from a live GET /v1beta/models call
against the key, not guessed. GroqProvider remains untested (no GROQ_API_KEY
available). Re-check available model IDs before a demo if this sits for a
while — these surfaces drift.

NOTE (2026-09-05): confirmed live via Google AI Studio's usage dashboard
that the free tier on gemini-3.5-flash-lite has both a per-minute (RPM)
and a per-day (RPD) cap, and we exceeded both (21/15 RPM, 501/500 RPD) —
distinct problems needing distinct fixes. RPM is fixed permanently by
pacing every real call to at least GEMINI_MIN_CALL_INTERVAL_SECONDS apart
(see below GeminiProvider — process-wide, not per-instance). RPD is a hard
wall for the rest of that day on that key; pacing cannot fix it. A 429 is
now classified (QuotaExceededError, distinct from a timeout/503) and, when
it's specifically the per-day quota, latches _gemini_daily_quota_exhausted
so every later call in this process fails instantly without hitting the
network — see the comments around both for the full reasoning. Swapping
GEMINI_API_KEY in .env and restarting the backend picks up the new key
with no other change needed: agent.py's load_dotenv(override=False) below
runs once per process at import time, GeminiProvider.__init__() reads
os.environ.get("GEMINI_API_KEY") fresh on every instance (get_provider()
constructs a new one per request), and a restart clears the in-memory
_gemini_daily_quota_exhausted latch along with everything else — the one
way this could fail to pick up a new key is if GEMINI_API_KEY is ALSO
exported as a real OS/shell environment variable (override=False means
that would win over .env); it isn't in this project's normal setup.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env (gitignored — holds GEMINI_API_KEY / GROQ_API_KEY locally) before
# anything below reads os.environ. Checks the repo root first (D:\Anchor\.env
# — where this project keeps it) and backend\.env second, so either location
# works. override=False: real process/shell env vars still win if both are
# set.
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env", override=False)
load_dotenv(_REPO_ROOT / "backend" / ".env", override=False)

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


# --- Free-tier rate limiting, shared by every GeminiProvider instance in
# this process --------------------------------------------------------------
#
# Confirmed live (2026-09-05, Google AI Studio usage dashboard) that
# gemini-3.5-flash-lite's free tier has two independent caps: ~15 requests
# per minute (RPM), and 500 requests per day (RPD). We were over both
# (21/15 RPM, 501/500 RPD). RPM is a pacing problem — permanently fixable
# in code. RPD is a hard wall for the rest of that day on that key —
# pacing can't fix it, only a fresh key (or the next day) can.
#
# GEMINI_MIN_CALL_INTERVAL_SECONDS: minimum gap enforced between the START
# of any two Gemini network calls, process-wide (a threading.Lock, not
# per-instance — get_provider() constructs a new GeminiProvider per
# request, so per-instance state wouldn't coordinate across concurrent
# requests). 60/15 = 4.0s is the exact RPM boundary; default 4.5s leaves
# margin (~13.3 req/min ceiling) for clock jitter. Env-overridable in case
# a different key/tier ever has a different cap.
GEMINI_MIN_CALL_INTERVAL_SECONDS = float(os.environ.get("GEMINI_MIN_CALL_INTERVAL_SECONDS", "4.5"))
_GEMINI_RATE_LOCK = threading.Lock()
_gemini_last_call_started_at = [0.0]  # 1-item list: mutable without `global`


def _gemini_throttle() -> None:
    """Block the calling thread just long enough that this call starts at
    least GEMINI_MIN_CALL_INTERVAL_SECONDS after the previous one started,
    process-wide. Holding the lock across the sleep is deliberate here —
    unlike a network call, this wait is bounded and known in advance
    (never more than GEMINI_MIN_CALL_INTERVAL_SECONDS), so it can't
    reproduce the WRITE_LOCK-held-across-an-unbounded-call bug (see
    app/db.py) — it's the opposite case: a lock that's SUPPOSED to
    serialize calls with a real, small, guaranteed-to-end wait."""
    with _GEMINI_RATE_LOCK:
        now = time.monotonic()
        wait = GEMINI_MIN_CALL_INTERVAL_SECONDS - (now - _gemini_last_call_started_at[0])
        if wait > 0:
            time.sleep(wait)
        _gemini_last_call_started_at[0] = time.monotonic()


# Latched for the lifetime of this process once a per-DAY quota exhaustion
# is confirmed (see _classify_quota_scope below) — every subsequent Gemini
# call, from any request, fails instantly without hitting the network at
# all, since we already know for certain it would just be another 429.
# Per-MINUTE exhaustion does NOT set this — that's transient (self-clears
# within a minute) and the pacing above is what prevents it recurring.
# Cleared only by restarting the process: swapping GEMINI_API_KEY in .env
# and restarting picks up a fresh key with a fresh quota (see get_provider()
# note), and Google's own daily counter also resets at a fixed clock time —
# either way, a restart is the reset path, matching how this key gets
# rotated in practice.
_gemini_daily_quota_exhausted = threading.Event()


class QuotaExceededError(RuntimeError):
    """Raised for a 429/RESOURCE_EXHAUSTED response — kept distinct from a
    transient network failure (timeout, 503) so it's recognizable as such
    in the audit trail (validation_note) rather than looking identical to
    a temporary blip. `scope` is best-effort: 'day' or 'minute' when
    Google's error body names which quota metric was hit (its documented
    QuotaFailure.violations[].quotaId format, e.g. containing
    'PerDayPerProject' or 'PerMinutePerProject'), else 'unknown'. Only
    'day' latches _gemini_daily_quota_exhausted — an 'unknown' 429 is
    treated as transient rather than risk incorrectly suppressing calls
    that would have succeeded again within a minute."""

    def __init__(self, message: str, scope: str):
        super().__init__(message)
        self.scope = scope


def _classify_quota_scope(exc: Exception) -> str:
    try:
        body = exc.response.text.lower()
    except Exception:  # noqa: BLE001 - response body isn't guaranteed text/decodable
        return "unknown"
    if "perday" in body or "per_day" in body or "requestsperday" in body:
        return "day"
    if "perminute" in body or "per_minute" in body or "requestsperminute" in body:
        return "minute"
    return "unknown"


class GeminiProvider(LLMProvider):
    """Google AI Studio Gemini API (free tier: Flash-Lite). Reads GEMINI_API_KEY.
    Model id via GEMINI_MODEL env var (default below) — check current Google AI
    Studio docs before relying on this id; it has not been verified live here."""
    name = "gemini"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, fallback_model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        # gemini-2.0-flash-lite 404'd against a live key on 2026-08-26. So did
        # gemini-2.5-flash-lite and gemini-2.5-flash: Google's error body
        # ("no longer available to new users... use models/gemini-3.5-flash-lite")
        # says dated/numbered snapshots get cut off from new API keys as newer
        # generations ship, which flips the usual pin-a-version advice — here
        # the rolling "-latest" alias is the one that kept working across that
        # cutoff. Pinned to gemini-3.5-flash-lite (the API's own recommended
        # replacement, confirmed 200 against this key) rather than
        # gemini-flash-lite-latest, so behavior doesn't silently shift versions
        # mid-demo; re-verify with GET /v1beta/models before relying on this
        # if it's been a while.
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        # Google has shipped several Flash-Lite generations in quick succession
        # this year — real risk the pinned model above gets cut again before
        # the demo. fallback_model defaults to the rolling "-latest" alias
        # specifically because it's immune to exactly this failure mode (it's
        # Google's job to keep it pointed at a live model, not ours to keep
        # re-pinning). One retry only — see decide() below.
        self.fallback_model = fallback_model or os.environ.get("GEMINI_MODEL_FALLBACK", "gemini-flash-lite-latest")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

    def _call(self, model: str, decision_input: DecisionInput):
        import requests

        if _gemini_daily_quota_exhausted.is_set():
            # Already confirmed exhausted this process's lifetime — don't
            # bother pacing or making the request, it can only 429 again.
            raise QuotaExceededError(
                "Gemini daily quota already confirmed exhausted this session (process); "
                "not attempting another call. Restart the backend after the quota resets "
                "or after swapping in a fresh GEMINI_API_KEY.",
                scope="day",
            )
        _gemini_throttle()  # never skip this: it's what keeps us under the RPM cap

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": json.dumps(decision_input.to_prompt_payload())}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
        }
        resp = requests.post(url, json=payload, timeout=20)
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                scope = _classify_quota_scope(exc)
                if scope == "day":
                    _gemini_daily_quota_exhausted.set()
                raise QuotaExceededError(
                    f"Gemini quota exceeded (scope={scope}): {exc}", scope=scope
                ) from exc
            raise
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text

    @staticmethod
    def _is_model_not_found(exc: Exception) -> bool:
        import requests
        return (
            isinstance(exc, requests.HTTPError)
            and exc.response is not None
            and exc.response.status_code == 404
        )

    def decide(self, decision_input: DecisionInput) -> AgentDecision:
        try:
            text = self._call(self.model, decision_input)
            model_used = self.model
        except Exception as exc:  # noqa: BLE001
            # QuotaExceededError deliberately falls straight through here:
            # it isn't a requests.HTTPError, so _is_model_not_found() is
            # False for it regardless of scope, and this re-raises without
            # attempting the fallback-model retry below — a quota problem
            # (per-minute OR per-day) is a project/key-level limit that
            # trying a different model id wouldn't fix, and ResilientProvider
            # still catches it and falls back to rule_based_fallback for
            # this decision either way.
            if not self._is_model_not_found(exc) or self.model == self.fallback_model:
                raise  # not a model-ID problem (or already on the fallback id) — let ResilientProvider handle it
            # One retry against the fallback model id (see __init__ note) before
            # giving up and letting ResilientProvider drop to rule_based_fallback.
            text = self._call(self.fallback_model, decision_input)
            model_used = self.fallback_model

        raw = _extract_json_object(text)
        decision = validate_decision(raw, decision_input, self.name, text)
        if model_used != self.model:
            note = f"Primary model {self.model!r} unavailable (404); served by fallback model {model_used!r}."
            decision.validation_note = f"{decision.validation_note} {note}" if decision.validation_note else note
        return decision


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


class ResilientProvider(LLMProvider):
    """
    Wraps a real LLM provider with automatic fallback to
    RuleBasedFallbackProvider if the primary call raises for any reason
    (network error, timeout, rate limit, malformed response, ...). This is
    the "safety net" — a live demo shouldn't die because an external API
    blipped once. It never disguises what happened: on fallback, the
    returned AgentDecision still carries provider="rule_based_fallback"
    (set by RuleBasedFallbackProvider itself) plus a validation_note
    recording the primary's failure, so the audit trail shows exactly which
    path produced the decision. The primary provider is not retried within
    a single call — one failure is enough to fall back for that decision.
    """
    def __init__(self, primary: LLMProvider, fallback: Optional[LLMProvider] = None):
        self.primary = primary
        self.fallback = fallback or RuleBasedFallbackProvider()
        self.name = primary.name

    def decide(self, decision_input: DecisionInput) -> AgentDecision:
        try:
            return self.primary.decide(decision_input)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any primary failure should fail over
            decision = self.fallback.decide(decision_input)
            decision.validation_note = (
                f"Primary provider {self.primary.name!r} raised {exc!r}; "
                f"used {self.fallback.name!r} safety net instead."
            )
            return decision


def get_provider(name: Optional[str] = None, resilient: bool = True) -> LLMProvider:
    """
    Provider selection, env-var driven so no code changes are needed to
    switch: LLM_PROVIDER=gemini|groq|rule_based_fallback. With no explicit
    name (and no LLM_PROVIDER env var), auto-detects from which API key is
    set (Gemini first, then Groq), falling back to the deterministic
    provider only if neither is configured.

    By default (resilient=True), a real LLM provider is wrapped in
    ResilientProvider so an API hiccup during a live demo degrades to the
    rule-based fallback instead of raising. Pass resilient=False to get the
    raw provider (e.g. to deliberately test failure handling, or to see a
    real exception instead of it being swallowed).
    """
    name = name or os.environ.get("LLM_PROVIDER")
    if name == "rule_based_fallback":
        return RuleBasedFallbackProvider()

    if name == "gemini":
        provider = GeminiProvider()
    elif name == "groq":
        provider = GroqProvider()
    elif os.environ.get("GEMINI_API_KEY"):
        provider = GeminiProvider()
    elif os.environ.get("GROQ_API_KEY"):
        provider = GroqProvider()
    else:
        return RuleBasedFallbackProvider()

    return ResilientProvider(provider) if resilient else provider
