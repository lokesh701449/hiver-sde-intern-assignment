"""
llm_agent.py
-------------
LLM agent for AmazonHelp intent classification + escalation decision via RAG.

Architecture:
  1. Receives customer message + conversation context
  2. Receives retrieved historical cases from ConversationRetriever
  3. Constructs a structured prompt
  4. Calls the LLM provider
  5. Returns a structured JSON response:
       {
         "intent": "...",
         "escalate": true/false,
         "escalation_reason": "...",
         "confidence": 0.0-1.0
       }

Provider selection (via LLM_PROVIDER env var):
  - "mock"      : deterministic rule-free mock (always available, no API key needed)
  - "openai"    : OpenAI API (requires OPENAI_API_KEY)
  - "anthropic" : Anthropic API (requires ANTHROPIC_API_KEY)
  - "gemini"    : Google Gemini API (requires GEMINI_API_KEY)

Changing the provider/model does NOT require modifying this file —
set the appropriate environment variables (see config.py).
"""

import json
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    ALL_INTENTS,
    LLM_PROVIDER, OPENAI_MODEL, ANTHROPIC_MODEL, GEMINI_MODEL,
    OLLAMA_MODEL, OLLAMA_BASE_URL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
)
from retrieval import ConversationCase
from escalation_policy import evaluate_escalation_policy


# ─── Structured output ────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    intent:            str
    escalate:          bool
    escalation_reason: str
    confidence:        float
    provider:          str
    latency_ms:        float
    reply:             str = ""
    llm_escalate:      Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "intent":            self.intent,
            "reply":             self.reply,
            "escalate":          self.escalate,
            "escalation_reason": self.escalation_reason,
            "llm_escalate":      self.llm_escalate,
            "confidence":        round(self.confidence, 3),
            "provider":          self.provider,
            "latency_ms":        round(self.latency_ms, 1),
        }

    def display(self):
        print(f"  Intent            : {self.intent}")
        if self.reply:
            print(f"  Reply             : {self.reply}")
        print(f"  Escalate          : {self.escalate}")
        print(f"  Escalation reason : {self.escalation_reason}")
        print(f"  Confidence        : {self.confidence:.3f}")
        print(f"  Provider          : {self.provider}")
        print(f"  Latency           : {self.latency_ms:.1f} ms")


# ─── Prompt builder ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AmazonHelp customer support analyst.

Your job is to analyze the CURRENT customer support conversation and evaluate TWO SEPARATE AND INDEPENDENT DECISIONS:
1. PRIMARY INTENT — exactly one of the 11 intents listed below.
2. ESCALATION DECISION — boolean (true or false) indicating whether the case requires escalation to a human agent, private lookup, or specialized investigation.

INTENT TAXONOMY (use exactly these strings):
- delivery_shipping_delay     : late, missing, or delayed shipments
- returns_refund_inquiry      : initiating returns, refund status, return window questions
- item_condition_issue        : damaged, defective, wrong item received
- device_hardware_support     : Kindle, Echo, Fire TV, tablet hardware or setup issues
- prime_membership_billing    : Prime subscription charges, membership questions
- digital_streaming_services  : Prime Video, Music, ebooks, digital content access
- account_access_security     : login problems, password reset, account locked, suspicious activity
- payment_promo_giftcard      : payment methods, gift cards, promotional codes, billing errors
- order_modification_cancellation : change or cancel an order before it ships
- marketplace_third_party_seller  : issues specifically with third-party (non-Amazon) sellers
- unclear_other               : cannot reliably determine a single intent from the above

INTENT CLASSIFICATION RULES:
- Classify the CURRENT customer conversation first based on the customer's actual primary problem.
- Use the full conversation context of the current case, not just the first message.
- Do not infer the current intent simply because most retrieved historical cases have similar wording.
- A customer mentioning payment while asking about a return → returns_refund_inquiry.
- A damaged item that was also late → item_condition_issue if damage is primary.
- Use unclear_other only when genuinely ambiguous, not as a default.

CRITICAL ESCALATION RULES (EVALUATE INDEPENDENTLY FROM INTENT):
1. Intent and escalation are separate decisions. Base escalation primarily on the current customer conversation and this explicit policy.
2. Do NOT default escalation to false. Think carefully whether private action or human intervention is needed.
3. ESCALATE (true) when: the request requires a private account, order, or customer lookup, or a sensitive action that a public support agent cannot perform publicly on social media.
4. ESCALATE (true) when: account/security cases involve compromised accounts, credentials, OTP/2FA, unauthorized access, password resets, or account recovery requiring identity verification.
5. ESCALATE (true) when: the request requires specific order/account investigation, tracking into private carrier systems, or exchanging private customer information.
6. ESCALATE (true) when: monetary or account-specific disputes (e.g. refund requests, billing adjustments, duplicate charges) require human agent verification or manual approval.
7. ESCALATE (true) when: marketplace or third-party seller disputes require Amazon intervention or claim processing.
8. DO NOT ESCALATE (false) merely because the customer is angry, frustrated, rude, or dissatisfied. Tone alone does NOT trigger escalation.
9. DO NOT ESCALATE (false) for ordinary delivery tracking questions, generic troubleshooting steps, general public policy questions, or simple self-service guidance answerable with public information.
10. Historical cases are FOR RESPONSE STYLE/RESOLUTION ONLY. They are NOT ground truth labels and must NOT determine escalation by themselves, as they contain no internal escalation labels.

OUTPUT FORMAT (strict JSON, no other text):
{
  "intent": "<one of the 11 intents>",
  "reply": "<short supportive public reply or empty string>",
  "escalate": <true|false>,
  "escalation_reason": "<concise concrete reason if true, or 'Not required' if false>",
  "confidence": <0.0 to 1.0>
}"""


def build_user_prompt(
    customer_message: str,
    conversation_context: str,
    retrieved_cases: List[ConversationCase],
) -> str:
    """Construct the user-turn prompt with 3 retrieved cases as RAG context."""
    lines = []

    # Current case
    lines.append("=== CURRENT CUSTOMER CONVERSATION ===")
    if conversation_context.strip():
        lines.append(f"Conversation context:\n{conversation_context.strip()}")
    lines.append(f"Customer message:\n{customer_message.strip()}")

    # Retrieved historical cases (reduced to top 3 and explicitly labeled)
    if retrieved_cases:
        lines.append("\n=== HISTORICAL EVIDENCE — FOR RESPONSE STYLE/RESOLUTION ONLY ===")
        lines.append("Note: The historical cases below are examples of how AmazonHelp handled similar situations.")
        lines.append("They are NOT labels, NOT ground truth, and must NOT determine intent or escalation by themselves.")
        for case in retrieved_cases[:3]:
            lines.append(f"\n[Historical Case #{case.rank} — similarity={case.score:.3f}]")
            lines.append(f"Customer: {case.customer_text[:300]}")
            if case.amazon_turns:
                lines.append(f"AmazonHelp replied: {case.amazon_turns[0][:200]}")
            else:
                lines.append("AmazonHelp: (no response on record)")
    else:
        lines.append("\n[No similar historical cases retrieved]")

    lines.append("\n=== YOUR TASK ===")
    lines.append("Classify the CURRENT customer conversation. Respond with ONLY the JSON object, no explanation.")
    return "\n".join(lines)


# ─── Provider interface ───────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Return the raw completion text."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class MockLLMProvider(LLMProvider):
    """
    Deterministic mock provider for testing the full pipeline without an API key.
    Uses simple keyword heuristics — intentionally not a good classifier.
    This is only for pipeline verification, not for evaluation.
    """

    @property
    def name(self) -> str:
        return "mock"

    def complete(self, system: str, user: str) -> str:
        # Extract the customer message from the prompt for simple keyword matching
        text = user.lower()

        # Rudimentary intent detection (not the real classifier)
        intent = "unclear_other"
        if any(w in text for w in ["deliver", "shipping", "shipment", "late", "arrived", "package", "parcel"]):
            intent = "delivery_shipping_delay"
        elif any(w in text for w in ["return", "refund", "money back", "send back"]):
            intent = "returns_refund_inquiry"
        elif any(w in text for w in ["damaged", "broken", "defective", "wrong item", "condition"]):
            intent = "item_condition_issue"
        elif any(w in text for w in ["kindle", "echo", "fire tv", "tablet", "device", "hardware"]):
            intent = "device_hardware_support"
        elif any(w in text for w in ["prime membership", "prime subscription", "prime billing", "prime charge"]):
            intent = "prime_membership_billing"
        elif any(w in text for w in ["prime video", "streaming", "music", "ebook", "digital"]):
            intent = "digital_streaming_services"
        elif any(w in text for w in ["account", "password", "login", "sign in", "locked", "security", "hacked"]):
            intent = "account_access_security"
        elif any(w in text for w in ["payment", "gift card", "promo", "coupon", "charged", "billing"]):
            intent = "payment_promo_giftcard"
        elif any(w in text for w in ["cancel", "modify", "change order", "order modification"]):
            intent = "order_modification_cancellation"
        elif any(w in text for w in ["seller", "third party", "marketplace", "vendor"]):
            intent = "marketplace_third_party_seller"

        # Escalation
        escalate = any(w in text for w in [
            "private", "sensitive", "account lookup", "investigation",
            "already tried", "several times", "dm", "direct message",
            "can't login", "locked out", "suspicious", "fraud"
        ])
        reason = (
            "Case requires private account lookup or sensitive action."
            if escalate else "Not required"
        )

        return json.dumps({
            "intent":            intent,
            "escalate":          escalate,
            "escalation_reason": reason,
            "confidence":        0.55 if intent != "unclear_other" else 0.30,
        })


class OpenAIProvider(LLMProvider):
    """OpenAI API provider. Requires OPENAI_API_KEY environment variable."""

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set.\n"
                "Export it or use LLM_PROVIDER=mock for local testing."
            )
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    @property
    def name(self) -> str:
        return f"openai/{OPENAI_MODEL}"

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic API provider. Requires ANTHROPIC_API_KEY environment variable."""

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is not set.\n"
                "Export it or use LLM_PROVIDER=mock for local testing."
            )
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    @property
    def name(self) -> str:
        return f"anthropic/{ANTHROPIC_MODEL}"

    def complete(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text



class GeminiProvider(LLMProvider):
    """
    Google Gemini provider using the google-genai SDK.
    Requires GEMINI_API_KEY environment variable.
    Model is configurable via GEMINI_MODEL env var (default: gemini-2.0-flash).
    Key is read from environment only — never hardcoded or logged.
    """

    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set.\n"
                "Run: export GEMINI_API_KEY=<your_key>  then retry."
            )
        try:
            from google import genai
            from google.genai import types as genai_types
            self._client = genai.Client(api_key=api_key)
            self._types  = genai_types
        except ImportError:
            raise ImportError(
                "google-genai package not installed.\n"
                "Run: pip install google-genai"
            )

    @property
    def name(self) -> str:
        return f"gemini/{GEMINI_MODEL}"

    def complete(self, system: str, user: str) -> str:
        """
        Call Gemini REST API directly via requests with a hard 45s timeout.

        Bypasses google-genai SDK's internal tenacity retry loop which hangs
        indefinitely on 503 UNAVAILABLE. We raise immediately on transient
        errors so evaluate_llm_rag.py's exponential-backoff retry takes over.
        """
        import requests

        api_key = os.environ.get("GEMINI_API_KEY", "")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": f"{system}\n\n{user}"}]}],
            "generationConfig": {
                "temperature":      LLM_TEMPERATURE,
                "maxOutputTokens":  LLM_MAX_TOKENS,
                "responseMimeType": "application/json",
            },
        }
        resp = requests.post(url, json=payload, timeout=45)
        if resp.status_code == 429:
            raise RuntimeError(f"429 RESOURCE_EXHAUSTED: {resp.text[:200]}")
        if resp.status_code == 503:
            raise RuntimeError(f"503 UNAVAILABLE: {resp.text[:200]}")
        if not resp.ok:
            raise RuntimeError(f"{resp.status_code} API error: {resp.text[:300]}")
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected Gemini response structure: {data}"
            ) from exc


class OllamaProvider(LLMProvider):
    """
    Ollama local API provider using HTTP POST requests.
    Requires local Ollama server running (default: http://localhost:11434).
    Model is configurable via OLLAMA_MODEL env var (default: qwen2.5:7b).
    Base URL is configurable via OLLAMA_BASE_URL env var (default: http://localhost:11434).
    """

    def __init__(self):
        self.model = OLLAMA_MODEL
        self.base_url = OLLAMA_BASE_URL.rstrip("/")

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    def complete(self, system: str, user: str) -> str:
        import requests

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": LLM_TEMPERATURE,
            }
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Ollama connection failed at {self.base_url}. "
                f"Is Ollama running locally? Error: {e}"
            ) from e

        if resp.status_code == 404:
            raise RuntimeError(
                f"Ollama model '{self.model}' not found on local server ({resp.status_code}). "
                f"Run: ollama pull {self.model}"
            )
        if not resp.ok:
            raise RuntimeError(f"Ollama API error ({resp.status_code}): {resp.text[:300]}")

        data = resp.json()
        try:
            content = data["message"]["content"]
            # Strip markdown code fences if model returned them
            content = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
            content = re.sub(r"\s*```$", "", content.strip()).strip()
            return content
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Ollama response structure: {data}") from exc


def get_provider(provider_name: str = None) -> LLMProvider:
    """Factory: return the appropriate LLM provider based on LLM_PROVIDER env var or explicit param."""
    name = (provider_name or LLM_PROVIDER).lower()
    if name == "openai":
        return OpenAIProvider()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "gemini":
        return GeminiProvider()
    if name == "ollama":
        return OllamaProvider()
    if name == "mock":
        return MockLLMProvider()
    raise ValueError(f"Unknown LLM_PROVIDER: {name!r}. Choose: mock | openai | anthropic | gemini | ollama")


# ─── Main agent ───────────────────────────────────────────────────────────────

class LLMAgent:
    """
    High-level agent: given a customer message + retrieved historical cases,
    calls the LLM and returns a structured ClassificationResult.
    """

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or get_provider()
        print(f"LLM provider: {self.provider.name}")

    def classify(
        self,
        customer_message: str,
        conversation_context: str = "",
        retrieved_cases: Optional[List[ConversationCase]] = None,
    ) -> ClassificationResult:
        """
        Run the full RAG-augmented classification pipeline.

        Returns a ClassificationResult with intent, escalation, reason, confidence.
        """
        user_prompt = build_user_prompt(
            customer_message,
            conversation_context,
            retrieved_cases or [],
        )

        t0 = time.time()
        raw = self.provider.complete(SYSTEM_PROMPT, user_prompt)
        latency_ms = (time.time() - t0) * 1000

        # Parse JSON from LLM (intent, reply, raw escalation prediction)
        result = self._parse_output(raw)
        llm_esc = result["escalate"]

        # Apply deterministic escalation policy
        policy_res = evaluate_escalation_policy(
            customer_message=customer_message,
            conversation_context=conversation_context,
            predicted_intent=result["intent"]
        )

        # Override escalation with deterministic policy decision, preserving raw LLM escalate for analysis
        result["escalate"]          = policy_res["escalate"]
        result["escalation_reason"] = policy_res["escalation_reason"]
        result["llm_escalate"]      = llm_esc
        result["latency_ms"]        = latency_ms
        result["provider"]          = self.provider.name
        return ClassificationResult(**result)

    def _parse_output(self, raw: str) -> dict:
        """Parse LLM output into a validated dict."""
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw.strip()).strip().strip("`")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON object from surrounding text
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}

        # Validate and normalise
        intent = data.get("intent", "unclear_other")
        if intent not in ALL_INTENTS:
            intent = "unclear_other"

        escalate = data.get("escalate", False)
        if isinstance(escalate, str):
            escalate = escalate.lower() in ("true", "yes", "1")

        return {
            "intent":            intent,
            "reply":             str(data.get("reply", "")),
            "escalate":          bool(escalate),
            "escalation_reason": str(data.get("escalation_reason", "Not required")),
            "confidence":        float(data.get("confidence", 0.5)),
        }
