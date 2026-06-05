from typing import Any, Dict, Optional, Sequence
import rich

from kbdebugger.compat.langchain import Document
from kbdebugger.utils.json import write_json
from kbdebugger.utils.time import now_utc_compact, now_utc_human
from .types import Qualities, SourceKind, DecomposeMode

def build_chunked_documents_payload(
    *,
    docs: Sequence[Document],
    created_at: str | None = None,
) -> dict[str, Any]:
    """
    Build a clean, JSON-serializable payload for chunked paragraph Documents.

    Design goals
    ------------
    - Avoid logging massive Docling internals (dl_meta.doc_items, bbox, prov, etc.)
    - Keep only what is useful for debugging + UI traceability:
        * top-level `source` once (if available)
        * per-doc `page_content`
        * per-doc `headings` (if available): metadata["dl_meta"]["headings"]

    Parameters
    ----------
    docs:
        Chunked LangChain Documents.
    created_at:
        Optional timestamp string; defaults to now_utc_human().

    Returns
    -------
    dict[str, Any]
        Clean payload ready for write_json(...) or API responses.
    """
    created_at = created_at or now_utc_human()

    # Source is repeated in every doc metadata; log it once if possible.
    source: str | None = None
    if docs:
        md0 = dict(getattr(docs[0], "metadata", {}) or {})
        source = md0.get("source")

    cleaned_docs: list[dict[str, Any]] = []
    for doc in docs:
        md = dict(getattr(doc, "metadata", {}) or {})
        dl_meta = md.get("dl_meta") or {}

        headings = None
        if isinstance(dl_meta, dict):
            headings_val = dl_meta.get("headings")
            if isinstance(headings_val, list) and headings_val:
                headings = headings_val

        cleaned_docs.append(
            {
                "page_content": getattr(doc, "page_content", "") or "",
                # Only include headings if we actually have them
                **({"headings": headings} if headings is not None else {}),
            }
        )

    payload: dict[str, Any] = {
        "source": source,
        "num_docs": len(docs),
        "docs": cleaned_docs,
        "created_at": created_at,
    }

    return payload


def save_chunked_documents_json(
    *,
    docs: list[Document],
    source_kind: SourceKind,
) -> dict[str, Any]:
    """
    Save chunked Documents to a clean JSON log for debugging/demo purposes.

    What we store
    -------------
    - source (once, top-level)
    - num_docs
    - docs: [{ page_content, headings? }]
    - created_at (human-friendly)

    Returns
    -------
    dict[str, Any]
        The payload (useful if caller wants to reuse it without rebuilding).
    """
    payload = build_chunked_documents_payload(docs=docs)

    # Keep compact time ONLY for filenames.
    created_at_compact = now_utc_compact()
    path = f"logs/01.1_chunker_output_docs_[{source_kind}]_{created_at_compact}.json"
    write_json(path, payload)

    rich.print(f"\n[INFO] Wrote chunker output log to {path}")
    return payload


def build_decomposer_payload(
    *,
    qualities: Qualities,
    quality_sources: Optional[Sequence[Dict[str, Any]]] = None,
    mode: DecomposeMode,
    num_input_docs: int,
    use_batch_decomposer: bool,
    batch_size: Optional[int],
    num_batches: Optional[int],
    parallel: bool,
    max_workers: Optional[int],
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a clean, JSON-serializable payload for the Decomposer stage.

    Notes
    -----
    - `created_at` is human-readable (for payloads/UI).
    - Filename timestamp stays compact (handled by logger).
    """
    created_at = created_at or now_utc_human()

    payload: Dict[str, Any] = {
        "created_at": created_at,
        # "mode": str(mode),  # or mode.value if it's an Enum
        "mode": mode.value if hasattr(mode, "value") else str(mode),
        "num_input_docs": num_input_docs,
        "num_output_qualities": len(qualities),
        "use_batch_decomposer": use_batch_decomposer,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "parallel": parallel,
        "max_workers": max_workers if parallel else None,
        "qualities": list(qualities),
    }

    if quality_sources is not None:
        payload["quality_sources"] = list(quality_sources)

    # Keep payload clean: drop None fields
    return {k: v for k, v in payload.items() if v is not None}


def save_qualities_json(
    *,
    qualities: Qualities,
    quality_sources: Optional[Sequence[Dict[str, Any]]] = None,
    mode: DecomposeMode,
    num_input_docs: int,
    use_batch_decomposer: bool,
    batch_size: Optional[int] = None,
    num_batches: Optional[int] = None,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    output_dir: str = "logs",
) -> Dict[str, Any]:
    """
    Save Decomposer output qualities to JSON and return the written payload.

    - Payload uses human-readable created_at.
    - Filename uses compact timestamp.
    """
    payload = build_decomposer_payload(
        qualities=qualities,
        quality_sources=quality_sources,
        mode=mode,
        num_input_docs=num_input_docs,
        use_batch_decomposer=use_batch_decomposer,
        batch_size=batch_size,
        num_batches=num_batches,
        parallel=parallel,
        max_workers=max_workers,
    )

    ts = now_utc_compact()
    path = f"{output_dir}/01.2_decomposer_qualities_{mode}_{ts}.json"
    write_json(path, payload)

    rich.print(f"\n[INFO] Wrote decomposer qualities log to {path}")
    return payload
