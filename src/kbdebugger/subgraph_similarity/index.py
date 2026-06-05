from __future__ import annotations

"""
Exact in-memory vector index backend.

The KG subgraph indexed by this pipeline is usually small, so exact NumPy search
is fast enough and much safer than native ANN libraries for the UI process.

This module keeps the public VectorIndex API used by the rest of the pipeline:
- create(dim=..., max_elements=...)
- add(vectors, items)
- search(query_vec, k)
- search_batch(query_vecs, k)
"""

from dataclasses import dataclass
from typing import Generic, List, Sequence, Tuple, TypeVar

import numpy as np

from .faiss_utils import (
    _as_float32_matrix,
    _as_float32_vector,
    _l2_normalize_rows,
)

T = TypeVar("T")


@dataclass
class VectorIndex(Generic[T]):
    """
    In-memory cosine index using exact NumPy matrix multiplication.

    Attributes
    ----------
    dim:
        Dimensionality of the embedding vectors.

    vectors:
        Normalized indexed vectors with shape (N, dim).

    payloads:
        List mapping vector row indexes to domain objects.
    """

    dim: int
    vectors: np.ndarray
    payloads: List[T]

    @classmethod
    def create(
        cls,
        *,
        dim: int,
        max_elements: int,
    ) -> "VectorIndex[T]":
        """
        Create an empty exact vector index.

        `max_elements` is retained for compatibility with previous backends.
        """
        del max_elements
        return cls(
            dim=dim,
            vectors=np.zeros((0, dim), dtype=np.float32),
            payloads=[],
        )

    def add(self, vectors: np.ndarray, items: Sequence[T]) -> None:
        """
        Add vectors and associated payload objects to the index.
        """
        matrix = _as_float32_matrix(vectors, name="vectors")

        if matrix.shape[1] != self.dim:
            raise ValueError(
                f"❌ Shape mismatch: expected vectors of shape (N, {self.dim}), got {matrix.shape}"
            )

        if len(matrix) != len(items):
            raise ValueError("❌ Number of vectors must match number of payload items.")

        matrix_normalized = _l2_normalize_rows(matrix)
        self.vectors = np.vstack([self.vectors, matrix_normalized]).astype(np.float32, copy=False)
        self.payloads.extend(list(items))

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[T, float]]:
        """
        Search for the k nearest neighbors of a single query vector.
        """
        if k <= 0:
            return []

        q = _as_float32_vector(query_vec, dim=self.dim, name="query_vec").reshape(1, -1)
        neighbors, scores = self.search_batch(q, k=k)

        out: List[Tuple[T, float]] = []
        row_neighbors = neighbors[0]
        row_scores = scores[0]

        n = min(len(row_neighbors), int(row_scores.shape[0]))
        for i in range(n):
            out.append((row_neighbors[i], float(row_scores[i])))

        return out

    def search_batch(self, query_vecs: np.ndarray, k: int) -> Tuple[
        List[List[T]],
        np.ndarray,
    ]:
        """
        Perform batched cosine-similarity search over the index.

        Returns
        -------
        (neighbors, scores):
            neighbors is a list of payload lists, one per query. It may contain
            fewer than k payloads when the index has fewer than k vectors.

            scores is always shape (Q, k), padded with 0.0 when needed.
        """
        q = _as_float32_matrix(query_vecs, name="query_vecs")

        if q.shape[1] != self.dim:
            raise ValueError(
                f"❌ Shape mismatch: expected query_vecs of shape (Q, {self.dim}), got {q.shape}"
            )

        num_q = int(q.shape[0])

        if k <= 0:
            return ([[] for _ in range(num_q)], np.zeros((num_q, 0), dtype=np.float32))

        if not self.payloads:
            return ([[] for _ in range(num_q)], np.zeros((num_q, k), dtype=np.float32))

        q_norm = np.ascontiguousarray(_l2_normalize_rows(q), dtype=np.float32)
        similarities = np.matmul(q_norm, self.vectors.T).astype(np.float32, copy=False)

        k_eff = min(k, len(self.payloads))
        top_ids = np.argsort(-similarities, axis=1)[:, :k_eff]
        top_scores = np.take_along_axis(similarities, top_ids, axis=1).astype(np.float32, copy=False)

        scores = np.zeros((num_q, k), dtype=np.float32)
        scores[:, :k_eff] = top_scores

        neighbors: List[List[T]] = []
        for row_ids in top_ids:
            neighbors.append([self.payloads[int(idx)] for idx in row_ids])

        return neighbors, scores
