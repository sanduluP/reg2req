from __future__ import annotations

import pytest

import kbdebugger.extraction.decompose as decompose_mod
from kbdebugger.compat.langchain import Document
from kbdebugger.extraction.types import DecomposeMode


def _docs(n: int) -> list[Document]:
    return [
        Document(page_content=f"Paragraph {i} about transparency.", metadata={"source": "doc.pdf"})
        for i in range(n)
    ]


def test_all_batches_failing_raises_clear_backend_error(monkeypatch):
    def boom(group):
        raise ConnectionError("Connection timed out after 8006 milliseconds")

    monkeypatch.setattr(decompose_mod, "_chunk_batch_to_qualities_decomposer", boom)

    with pytest.raises(RuntimeError) as excinfo:
        decompose_mod.decompose_documents(
            _docs(4),
            mode=DecomposeMode.CHUNKS,
            batch_size=2,
            use_batch_decomposer=True,
            parallel=False,
        )

    message = str(excinfo.value)
    assert "failed for all 2 batch(es)" in message
    assert "MODEL_BACKEND" in message
    assert "Connection timed out" in message


def test_all_batches_failing_raises_in_parallel_mode_too(monkeypatch):
    def boom(group):
        raise ConnectionError("backend down")

    monkeypatch.setattr(decompose_mod, "_chunk_batch_to_qualities_decomposer", boom)

    with pytest.raises(RuntimeError, match="failed for all 2 batch"):
        decompose_mod.decompose_documents(
            _docs(4),
            mode=DecomposeMode.CHUNKS,
            batch_size=2,
            use_batch_decomposer=True,
            parallel=True,
            max_workers=2,
        )


def test_partial_batch_failure_stays_best_effort(monkeypatch):
    calls = {"n": 0}

    def flaky(group):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transient failure")
        return [[f"Quality from: {text[:20]}"] for text in group]

    monkeypatch.setattr(decompose_mod, "_chunk_batch_to_qualities_decomposer", flaky)

    qualities, log_payload = decompose_mod.decompose_documents(
        _docs(4),
        mode=DecomposeMode.CHUNKS,
        batch_size=2,
        use_batch_decomposer=True,
        parallel=False,
    )

    # Second batch succeeded → no raise, its qualities are kept.
    assert len(qualities) == 2
    # The partial failure is visible in the log payload for debugging.
    assert log_payload["num_failed_batches"] == 1
    assert any("transient failure" in m for m in log_payload["batch_failure_messages"])


def test_legitimately_empty_output_does_not_raise(monkeypatch):
    def empty(group):
        return [[] for _ in group]

    monkeypatch.setattr(decompose_mod, "_chunk_batch_to_qualities_decomposer", empty)

    qualities, log_payload = decompose_mod.decompose_documents(
        _docs(4),
        mode=DecomposeMode.CHUNKS,
        batch_size=2,
        use_batch_decomposer=True,
        parallel=False,
    )

    assert qualities == []
    assert "num_failed_batches" not in log_payload
