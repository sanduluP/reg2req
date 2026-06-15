from __future__ import annotations

import math
import re
from typing import Any, Sequence

from kbdebugger.compat.langchain import Document


REFERENCE_HEADINGS = {
    "references",
    "reference",
    "bibliography",
    "works cited",
    "literature cited",
    "cited literature",
    "references and notes",
    "notes and references",
    "reference list",
    "bibliographical references",
}

_SECTION_PREFIX_RE = re.compile(
    r"^\s*(?:(?:section|chapter)\s+)?"
    r"(?:(?:\d+(?:\.\d+)*)|(?:[ivxlcdm]+)|(?:[a-z]))"
    r"(?:[\.\):\-])?\s+",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")


def normalize_reference_heading(text: str) -> str:
    """Normalize a potential section heading for exact reference-heading matching."""
    value = str(text or "").strip()
    if not value:
        return ""

    # Headings occasionally include markdown/list markers or trailing punctuation.
    value = value.lstrip("#*•-–— ").strip()
    value = _SECTION_PREFIX_RE.sub("", value).strip()
    value = value.strip(" .:;,-–—()[]{}")
    value = value.replace("&", " and ")
    value = _SPACE_RE.sub(" ", value).strip().lower()
    return value


def is_reference_heading(text: str) -> bool:
    return normalize_reference_heading(text) in REFERENCE_HEADINGS


def _doc_headings(doc: Document) -> list[str]:
    metadata = getattr(doc, "metadata", {}) or {}
    dl_meta = metadata.get("dl_meta") if isinstance(metadata, dict) else None

    candidates: list[str] = []
    if isinstance(dl_meta, dict):
        headings = dl_meta.get("headings")
        if isinstance(headings, list):
            candidates.extend(str(heading) for heading in headings if str(heading).strip())

    if isinstance(metadata, dict):
        headings = metadata.get("headings")
        if isinstance(headings, list):
            candidates.extend(str(heading) for heading in headings if str(heading).strip())

    return candidates


def _first_short_line(text: str, *, max_chars: int = 90) -> str | None:
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if len(line) <= max_chars:
            return line
        return None
    return None


def _reference_heading_candidates(doc: Document) -> list[str]:
    candidates = _doc_headings(doc)
    first_line = _first_short_line(getattr(doc, "page_content", "") or "")
    if first_line:
        candidates.append(first_line)
    return candidates


def _empty_metadata(*, enabled: bool, mode: str, before: int, after: int | None = None) -> dict[str, Any]:
    after_count = before if after is None else after
    return {
        "reference_filter_enabled": enabled,
        "reference_filter_mode": mode,
        "reference_section_detected": False,
        "trigger_heading": None,
        "trigger_doc_index": None,
        "num_docs_before_filter": before,
        "num_docs_after_filter": after_count,
        "num_reference_docs_removed": before - after_count,
    }


def filter_reference_section(
    docs: Sequence[Document],
    *,
    mode: str = "conservative",
    enabled: bool = True,
) -> tuple[list[Document], dict[str, Any]]:
    """Drop a late References/Bibliography section from Docling paragraph documents."""
    doc_list = list(docs)
    before = len(doc_list)
    normalized_mode = str(mode or "conservative").strip().lower()

    if not enabled:
        return doc_list, _empty_metadata(enabled=False, mode=normalized_mode, before=before)

    if normalized_mode != "conservative":
        raise ValueError(f"Unsupported reference section filter mode: {mode!r}")

    if not doc_list:
        return doc_list, _empty_metadata(enabled=True, mode=normalized_mode, before=before)

    min_trigger_index = max(5, math.ceil(before * 0.45))

    for idx, doc in enumerate(doc_list):
        if idx < min_trigger_index:
            continue

        for candidate in _reference_heading_candidates(doc):
            if is_reference_heading(candidate):
                filtered = doc_list[:idx]
                metadata = {
                    "reference_filter_enabled": True,
                    "reference_filter_mode": normalized_mode,
                    "reference_section_detected": True,
                    "trigger_heading": candidate.strip(),
                    "trigger_doc_index": idx,
                    "num_docs_before_filter": before,
                    "num_docs_after_filter": len(filtered),
                    "num_reference_docs_removed": before - len(filtered),
                }
                return filtered, metadata

    return doc_list, _empty_metadata(enabled=True, mode=normalized_mode, before=before)
