from __future__ import annotations
from dotenv import load_dotenv

from dataclasses import dataclass
from typing import Any, NoReturn, Final
import json
import os
import time

import requests
from .groq_responder import GroqResponder
from .llm_protocol import LLMResponder


# -----------------------------
# Environment / config
# -----------------------------
load_dotenv(override=True)

# Select backend: "http" (default) or "hf_local"
MODEL_BACKEND: Final[str] = os.getenv("MODEL_BACKEND", "http").lower()

# HTTP backend config (OpenAI/compatible chat completions)
MODEL_SERVICE_URL: Final[str] = os.getenv(
    "MODEL_SERVICE_URL",
    "http://serv-3306.kl.dfki.de:8000/v1/chat/completions",
)

MODEL_SERVICE_NAME: Final[str] = os.getenv(
    "MODEL_SERVICE_NAME",
    "qwen3.6:27b",
    # "llama3.3-70b-instruct-fp8",
)

REQUEST_TIMEOUT: Final[float] = float(os.getenv("REQUEST_TIMEOUT", "30.0")) # seconds
REQUEST_RETRIES: Final[int] = int(os.getenv("REQUEST_RETRIES", "2"))

# Optional bearer token for hosted OpenAI-compatible APIs (e.g. DeepSeek, OpenAI).
# Empty means no Authorization header is sent (works for unauthenticated
# internal services such as the DFKI model server).
MODEL_API_KEY: Final[str] = os.getenv("MODEL_API_KEY", "").strip()


def _load_model_service_extra_body() -> dict[str, Any]:
    raw = os.getenv("MODEL_SERVICE_EXTRA_BODY_JSON", "").strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[HTTPChatResponder] Ignoring invalid MODEL_SERVICE_EXTRA_BODY_JSON: {exc}")
        return {}

    if not isinstance(parsed, dict):
        print("[HTTPChatResponder] Ignoring MODEL_SERVICE_EXTRA_BODY_JSON because it is not a JSON object.")
        return {}

    return parsed


MODEL_SERVICE_EXTRA_BODY: Final[dict[str, Any]] = _load_model_service_extra_body()

JSON_SYSTEM_MESSAGE: Final[str] = (
    "You are a JSON-only API. Return exactly one JSON object and no other text. "
    "Do not include markdown, comments, explanations, or reasoning. "
    "The response must start with '{' and end with '}'."
)

# HF local fallback model (CPU-friendly)
HF_LOCAL_MODEL: Final[str] = os.getenv(
    "HF_LOCAL_MODEL",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)
HF_DEVICE: Final[str] = os.getenv("HF_DEVICE", "cpu")  # e.g., "cpu" or "cuda"
HF_MAX_NEW_TOKENS: Final[int] = int(os.getenv("HF_MAX_NEW_TOKENS", "256"))


# -----------------------------
# HTTP (OpenAI-compatible) client
# -----------------------------
def _coerce_text(value: Any) -> str:
    """
    Normalize text fields used by OpenAI-compatible and compatible-ish APIs.

    OpenAI chat responses usually return a string in message.content, but some
    APIs return a list of typed content parts.
    """
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content", "value"):
                    text = item.get(key)
                    if isinstance(text, str):
                        parts.append(text)
                        break
        return "".join(parts).strip()

    return ""


def _extract_assistant_content(payload: Any) -> str:
    """
    Extract final assistant text from several common LLM response envelopes.

    Supported shapes include:
    - OpenAI-compatible chat: choices[0].message.content
    - OpenAI legacy completions: choices[0].text
    - Ollama native generate/chat-like payloads: response, message.content
    - Anthropic-like payloads: content text parts

    Reasoning fields are intentionally not returned: they are not final answers
    and often contain non-JSON chain-of-thought text.
    """
    if not isinstance(payload, dict):
        return ""

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                content = _coerce_text(message.get("content"))
                if content:
                    return content

            content = _coerce_text(choice.get("text"))
            if content:
                return content

            delta = choice.get("delta")
            if isinstance(delta, dict):
                content = _coerce_text(delta.get("content"))
                if content:
                    return content

    message = payload.get("message")
    if isinstance(message, dict):
        content = _coerce_text(message.get("content"))
        if content:
            return content

    for key in ("response", "output_text", "text", "content"):
        content = _coerce_text(payload.get(key))
        if content:
            return content

    return ""


def _extract_reasoning_preview(payload: Any, limit: int = 400) -> str:
    """
    Return a short preview of reasoning-only output for diagnostics.
    """
    if not isinstance(payload, dict):
        return ""

    values: list[str] = []
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                for key in ("reasoning", "reasoning_content"):
                    text = _coerce_text(message.get(key))
                    if text:
                        values.append(text)
            delta = choice.get("delta")
            if isinstance(delta, dict):
                for key in ("reasoning", "reasoning_content"):
                    text = _coerce_text(delta.get(key))
                    if text:
                        values.append(text)

    for key in ("reasoning", "reasoning_content"):
        text = _coerce_text(payload.get(key))
        if text:
            values.append(text)

    preview = " ".join(values).strip()
    return preview[:limit]


def _finish_reason(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    reason = choice.get("finish_reason")
    return reason if isinstance(reason, str) else ""


@dataclass
class HTTPChatResponder:
    """
    Calls an OpenAI-compatible /v1/chat/completions endpoint.

    Expected inputs to .invoke():
    {
        "prompt": "<final prompt string>",   # required
        "max_tokens": 500,                   # optional override
        "temperature": 0.2,                  # optional
        ...
    }

    Returns assistant message content as a string.
    """
    url: str
    model: str
    timeout: float = 60.0
    retries: int = 2
    api_key: str = ""

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def invoke(self, inputs: dict[str, Any]) -> str:
        prompt = inputs.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("HTTPChatResponder.invoke expects inputs['prompt'] as a non-empty string.")

        json_mode = bool(inputs.get("json_mode", False))
        messages = []
        if json_mode:
            messages.append({"role": "system", "content": JSON_SYSTEM_MESSAGE})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(inputs.get("max_tokens", 500)),
        }
        if "temperature" in inputs:
            data["temperature"] = float(inputs["temperature"])
        if json_mode:
            data["response_format"] = {"type": "json_object"}
        if MODEL_SERVICE_EXTRA_BODY:
            data.update(MODEL_SERVICE_EXTRA_BODY)

        last_exception: Exception | None = None
        for attempt in range(1, self.retries + 2):  # first try + retries
            started = time.perf_counter()
            try:
                resp = requests.post(self.url, json=data, headers=self._headers(), timeout=self.timeout)
                if resp.status_code == 400 and "response_format" in data:
                    # Some OpenAI-compatible servers do not support structured
                    # output parameters. Keep the JSON-only system prompt and
                    # retry without the incompatible request field.
                    retry_data = dict(data)
                    retry_data.pop("response_format", None)
                    resp = requests.post(self.url, json=retry_data, headers=self._headers(), timeout=self.timeout)
                resp.raise_for_status() # Raises HTTPError, if one occurred.
                payload = resp.json()
                content = _extract_assistant_content(payload)
                if not content:
                    reasoning_preview = _extract_reasoning_preview(payload)
                    finish_reason = _finish_reason(payload)
                    details = []
                    if finish_reason:
                        details.append(f"finish_reason={finish_reason!r}")
                    if reasoning_preview:
                        details.append(f"reasoning_preview={reasoning_preview!r}")
                    detail_text = f" ({', '.join(details)})" if details else ""
                    raise ValueError(f"LLM response did not contain final assistant content{detail_text}.")
                elapsed = time.perf_counter() - started
                print(
                    "[HTTPChatResponder] "
                    f"model={self.model} prompt_chars={len(prompt)} max_tokens={data.get('max_tokens')} "
                    f"elapsed_s={elapsed:.2f} retries_used={attempt - 1} failures={attempt - 1}"
                )
                return content
            except Exception as exc:
                last_exception = exc
                if attempt <= self.retries:
                    time.sleep(0.5 * attempt)
                    continue
                else:
                    raise RuntimeError(f"HTTPChatResponder failed after {attempt} attempts: {exc}") from exc
                
        # It should not reach here, but mypy needs a return
        return ""
        
        # # This should never be reached due to the raise above, but added for type safety
        # raise RuntimeError(f"HTTPChatResponder failed after all attempts: {last_exception}")


# -----------------------------
# HF local client (simple text-generation)
# -----------------------------
class HFLocalResponder:
    """
    Minimal local Hugging Face text-generation backend.
    Loads model/tokenizer lazily on first call. Works on 🚗 CPU by default.

    Expected inputs to .invoke():
    {
        "prompt": "<final prompt string>",   # required
        "max_tokens": 256,                   # optional override
        ...
    }

    Returns plain generated text string (no special chat formatting).
    """
    def __init__(self, model_name: str, device: str = "cpu", max_new_tokens: int = 256) -> None:
        self.model_name = model_name
        self.device = device
        self.default_max_new_tokens = max_new_tokens
        self._pipe = None  # lazy init

    def _ensure_pipeline(self):
        if self._pipe is not None:
            return
        # Lazy import to avoid heavy deps until needed
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.device == "cuda" else -1,
        )

    def invoke(self, inputs: dict[str, Any]) -> str:
        prompt = inputs.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("HFLocalResponder.invoke expects inputs['prompt'] as a non-empty string.")
        max_new_tokens = int(inputs.get("max_tokens", self.default_max_new_tokens))
        self._ensure_pipeline()

        # Generate; pipeline returns list[dict]
        outs = self._pipe(prompt, max_new_tokens=max_new_tokens)  # type: ignore[attr-defined]
        if isinstance(outs, list) and outs and "generated_text" in outs[0]:
            return outs[0]["generated_text"]
        # Fallback stringify
        return str(outs)


def _unsupported_backend(backend: str) -> NoReturn:
    raise ValueError(f"Unsupported MODEL_BACKEND: {backend!r}")

# -----------------------------
# Factory
# -----------------------------
def get_llm_responder() -> LLMResponder:
    """
    Factory that returns an object satisfying LLMResponder.
    Chooses backend via MODEL_BACKEND: "groq", "http", or "hf_local".

    Usage:
        llm = get_llm_responder()
        result = llm.invoke({"prompt": "Hello model!"})
    """
    backend = os.getenv("MODEL_BACKEND", "http").lower()

    match backend:
        case "groq":
            return GroqResponder(model=os.getenv("GROQ_MODEL"))
        case "hf_local":
            return HFLocalResponder(
                model_name=HF_LOCAL_MODEL,
                device=HF_DEVICE,
                max_new_tokens=HF_MAX_NEW_TOKENS,
            )
        case "http":
            return HTTPChatResponder(
                url=MODEL_SERVICE_URL,
                model=MODEL_SERVICE_NAME,
                timeout=REQUEST_TIMEOUT,
                retries=REQUEST_RETRIES,
                api_key=MODEL_API_KEY,
            )
        case _:
            _unsupported_backend(backend)  # NoReturn → type checker knows we never return here


# -----------------------------
# Convenience wrapper (optional)
# -----------------------------
def respond(prompt: str, **kwargs: Any) -> str:
    """
    Convenience function for one-off calls without importing the responder:
        respond("your final prompt string", max_tokens=200)

    Equivalent to:
        get_llm_responder().invoke({"prompt": prompt, "max_tokens": 200})
    """
    llm = get_llm_responder()
    payload: dict[str, Any] = {"prompt": prompt}
    payload.update(kwargs)
    response = llm.invoke(payload)
    return response
