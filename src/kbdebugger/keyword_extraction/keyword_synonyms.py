from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import rich

from kbdebugger.llm.model_access import respond
from kbdebugger.prompts import render_prompt
from kbdebugger.utils import ensure_json_object


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RUNTIME_CACHE_PATH = "runtime/keyword_synonyms_cache.json"
_DEFAULT_CURATED_DEFAULTS_PATH = "data/keyword_synonyms.json"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_keyword(keyword: str) -> str:
    return " ".join(str(keyword or "").lower().split())


def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return _PROJECT_ROOT / resolved


def _synonym_cache_enabled(value: bool | None) -> bool:
    return _env_bool("KB_KEYWORD_SYNONYM_CACHE_ENABLED", True) if value is None else bool(value)


def _synonym_cache_write_enabled(value: bool | None) -> bool:
    return _env_bool("KB_KEYWORD_SYNONYM_CACHE_WRITE", True) if value is None else bool(value)


def _synonym_cache_path(value: str | Path | None) -> str:
    raw = str(value or os.getenv("KB_KEYWORD_SYNONYM_CACHE_PATH", _DEFAULT_RUNTIME_CACHE_PATH)).strip()
    return raw or _DEFAULT_RUNTIME_CACHE_PATH


def _synonym_defaults_path(value: str | Path | None) -> str:
    raw = str(value or os.getenv("KB_KEYWORD_SYNONYM_DEFAULTS_PATH", _DEFAULT_CURATED_DEFAULTS_PATH)).strip()
    return raw or _DEFAULT_CURATED_DEFAULTS_PATH


def _dedupe_limit(values: list[str] | tuple[str, ...], *, keyword: str, limit: int = 10) -> tuple[str, ...]:
    seen = {keyword}
    clean: list[str] = []
    for value in values:
        term = " ".join(str(value or "").strip().split())
        key = term.lower()
        if not term or key in seen:
            continue
        seen.add(key)
        clean.append(term)
        if len(clean) >= limit:
            break
    return tuple(clean)


def _read_synonym_file(path: str | Path, *, label: str) -> dict[str, tuple[str, ...]]:
    resolved = _resolve_project_path(path)
    if not resolved.exists():
        return {}

    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - cache must never abort the pipeline
        rich.print(
            f"[yellow][Keyword Synonyms][/yellow] Could not read {label} file "
            f"{str(resolved)!r}; ignoring it. Reason: {exc}"
        )
        return {}

    if not isinstance(raw, Mapping):
        rich.print(
            f"[yellow][Keyword Synonyms][/yellow] Expected {label} file "
            f"{str(resolved)!r} to contain a JSON object; ignoring it."
        )
        return {}

    out: dict[str, tuple[str, ...]] = {}
    for raw_keyword, raw_values in raw.items():
        keyword = _normalize_keyword(str(raw_keyword))
        if not keyword:
            continue
        if not isinstance(raw_values, list):
            continue
        synonyms = _dedupe_limit(raw_values, keyword=keyword)
        if synonyms:
            out[keyword] = synonyms
    return out


def _write_runtime_cache(path: str | Path, *, keyword: str, synonyms: tuple[str, ...]) -> None:
    resolved = _resolve_project_path(path)
    try:
        existing = _read_synonym_file(resolved, label="runtime synonym cache")
        existing[keyword] = synonyms
        serializable = {key: list(values) for key, values in sorted(existing.items())}

        resolved.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = resolved.with_name(f".{resolved.name}.tmp")
        tmp_path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(resolved)
    except Exception as exc:  # noqa: BLE001 - cache writes are opportunistic
        rich.print(
            f"[yellow][Keyword Synonyms][/yellow] Could not write runtime synonym cache "
            f"{str(resolved)!r}. Reason: {exc}"
        )


def _morphology_fallback_synonyms(keyword: str) -> tuple[str, ...]:
    variants: list[str] = []
    if keyword.endswith("ability"):
        stem = keyword[:-7]
        variants.extend([stem + "able", stem + "ation", stem])
    elif keyword.endswith("ibility"):
        stem = keyword[:-7]
        variants.extend([stem + "ible", stem + "ion", stem])
    elif keyword.endswith("ness"):
        variants.append(keyword[:-4])
    elif keyword.endswith("ity"):
        variants.append(keyword[:-3])
    elif keyword.endswith("tion"):
        variants.append(keyword[:-3] + "e")
    elif keyword.endswith("ment"):
        variants.append(keyword[:-4])

    return _dedupe_limit(tuple(variants), keyword=keyword)


@lru_cache(maxsize=256)
def _generate_synonyms_for_normalized_keyword(
    keyword: str,
    cache_enabled: bool = True,
    cache_path: str = _DEFAULT_RUNTIME_CACHE_PATH,
    defaults_path: str = _DEFAULT_CURATED_DEFAULTS_PATH,
    cache_write: bool = True,
) -> tuple[str, ...]:
    if cache_enabled:
        runtime_cache = _read_synonym_file(cache_path, label="runtime synonym cache")
        cached = runtime_cache.get(keyword)
        if cached:
            rich.print("[green][Keyword Synonyms][/green] Loaded cached synonyms:", list(cached))
            return cached

    curated_defaults = _read_synonym_file(defaults_path, label="curated synonym defaults")
    curated = curated_defaults.get(keyword)
    if curated:
        rich.print("[green][Keyword Synonyms][/green] Loaded curated synonyms:", list(curated))
        return curated

    prompt = render_prompt("keyword_synonyms", keyword=keyword)

    try:
        raw = respond(prompt, json_mode=True, temperature=0.0, max_tokens=1024)
        obj = ensure_json_object(raw)
        synonyms = obj.get("synonyms", [])
        if not isinstance(synonyms, list):
            synonyms = []
        normalized = _dedupe_limit(synonyms, keyword=keyword)
        if not normalized:
            normalized = _morphology_fallback_synonyms(keyword)
        if normalized and cache_enabled and cache_write:
            _write_runtime_cache(cache_path, keyword=keyword, synonyms=normalized)
        rich.print("[green][LLM Synonym Generation][/green] Generated synonyms:", list(normalized))
        return normalized
    except Exception as exc:  # noqa: BLE001 - keyword expansion must not abort the pipeline
        fallback = _morphology_fallback_synonyms(keyword)
        rich.print(
            "[yellow][LLM Synonym Generation][/yellow] "
            f"Falling back to local synonyms for {keyword!r}: {list(fallback)} "
            f"(reason: {exc})"
        )
        return fallback


def generate_synonyms_for_keyword(
    keyword: str,
    *,
    cache_enabled: bool | None = None,
    cache_path: str | Path | None = None,
    defaults_path: str | Path | None = None,
    cache_write: bool | None = None,
) -> list[str]:
    """
    Return up to 10 keyword-expansion terms using persistent local caching.

    Lookup order:
    1. in-process LRU cache
    2. runtime JSON cache
    3. curated JSON defaults
    4. LLM generation, written back to the runtime cache
    5. local morphology fallback if LLM generation fails
    """
    normalized_keyword = _normalize_keyword(keyword)
    if not normalized_keyword:
        return []

    resolved_cache_enabled = _synonym_cache_enabled(cache_enabled)
    resolved_cache_write = _synonym_cache_write_enabled(cache_write)
    resolved_cache_path = _synonym_cache_path(cache_path)
    resolved_defaults_path = _synonym_defaults_path(defaults_path)

    return list(
        _generate_synonyms_for_normalized_keyword(
            normalized_keyword,
            resolved_cache_enabled,
            resolved_cache_path,
            resolved_defaults_path,
            resolved_cache_write,
        )
    )
