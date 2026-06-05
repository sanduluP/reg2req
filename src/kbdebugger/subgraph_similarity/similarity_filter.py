from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from kbdebugger.subgraph_similarity.logging import build_qualities_to_subgraph_similarity_payload
from kbdebugger.types.ui import ProgressCallback
import numpy as np

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

from kbdebugger.types import GraphRelation
from kbdebugger.utils.progress import stage_status
from .encoder import TextEncoder
from .index import VectorIndex
from .types import DroppedQuality, KeptQuality, Quality, SubgraphSimilarityFilterConfig
from kbdebugger.utils.json import write_json
from kbdebugger.utils.time import now_utc_compact

"""
Vector similarity filter for *candidate qualities* (atomic sentences).

What changed (important)
-----------------------
Vector similarity does not actually require structured triplets — it requires semantic meaning.
That meaning is best represented by the *quality sentence* produced by the
Decomposer module.

So in this pipeline, we do:

    1) Decomposer LLM -> Qualities (atomic sentences)
    2) Graph Retriever -> KG subgraph relations (each has r.sentence)
    3) SubgraphSimilarityFilter -> compare qualities vs KG sentences
    4) Triplet extractor LLM -> run ONLY on the kept qualities

This change reduces LLM load and improves semantic matching quality.

Core idea
---------
- Build an in-memory *vector index* over KG subgraph sentences (r.sentence).
- Embed each *quality sentence* as a query vector and run cosine similarity search over the index.
- Drop quality if max similarity < threshold. e.g., 0.7
- Keep quality and store its top-k nearest KG relations as "context".

This module is intentionally:
- pure (no I/O, no DB writes)
- deterministic (given encoder + inputs)
- easy to test

Dependencies
------------
This module depends on:
- kbdebugger.subgraph_similarity.encoder.TextEncoder (pluggable embedding model)
- kbdebugger.subgraph_similarity.index.VectorIndex (pluggable vector index backend; HNSW in MVP)
"""

# ---------------------------------------------------------------------------
# Text shaping helpers
# ---------------------------------------------------------------------------
def quality_to_text(q: Quality) -> str:
    """
    Convert a quality to the text used for embedding.

    For now this is simply the quality itself.

    Why not add prefixes like "claim: ..."?
    --------------------------------------
    You *can*, but it changes embedding behavior. For an MVP, keeping it raw is
    simplest and most faithful to the decomposer output.
    """
    return q


def relation_to_text(r: GraphRelation) -> str:
    """
    Convert a KG relation into a natural-language sentence suitable for embedding.

    Design goals
    ------------
    - Produce text that is stylistically similar to decomposer "qualities"
      (plain English sentences).
    - Avoid structured metadata prefixes like "Source:", "Predicate:", etc.,
      because qualities are embedded as raw sentences.
    - Be robust to missing properties.

    Strategy
    --------
    1) Prefer an existing provenance sentence (r.edge.properties["sentence"]).
       This is assumed to be the strongest semantic signal.
    2) Normalize snake_case tokens into natural language.
    3) If no sentence is available, synthesize a simple natural sentence:
           "<source> <predicate phrase> <target>"

    Examples
    --------
    Input relation:
        bias --is_threat_to--> fairness

    Output text:
        "Bias is a threat to fairness."

    Returns
    -------
    str
        Natural-language sentence used for embedding.
    """

    def _humanize(token: str) -> str:
        """
        Convert a snake_case or underscored token into readable text.

        Example:
            "is_subclass_of" -> "is subclass of"
            "socioeconomic_status" -> "socioeconomic status"
        """
        return token.replace("_", " ").strip()

    def _capitalize_sentence(text: str) -> str:
        """
        Capitalize the first letter of a sentence.
        Example:
            "this is a sentence." -> "This is a sentence."
        """
        return text[:1].upper() + text[1:]

    props = r["edge"]["properties"]

    # 1) Preferred: use stored sentence if available
    sentence = props.get("sentence")
    if sentence:
        # Normalize snake_case inside the sentence
        text = _humanize(str(sentence))
        return _capitalize_sentence(text)

    # 2) Fallback: synthesize a sentence from S-P-O
    source = _humanize(r["source"]["label"])
    target = _humanize(r["target"]["label"])
    predicate = _humanize(r["edge"]["label"])

    text = f"{source} {predicate} {target}"
    return _capitalize_sentence(text)


# ---------------------------------------------------------------------------
# Main component
# ---------------------------------------------------------------------------
@dataclass
class SubgraphSimilarityFilter:
    """
    Filter Decomposer-produced qualities using similarity to a KG subgraph.

    Parameters
    ----------
    encoder:
        A TextEncoder (embedding model wrapper). Must be used consistently for:
            - KG relation sentences
            - candidate qualities

        If different encoders are used, vector similarity becomes meaningless.

    top_k:
        How many nearest KG relations to retrieve per quality (i.e., query vector).
        These are preserved as context for later semantic verification.

        Typical values:
            3-10

    threshold:
        Minimum cosine similarity required for a quality to be kept.

        Decision rule:
            - Compute k nearest neighbor similarity scores for each quality (i.e., query vector).
            - max_score = max(scores)
            - if max_score < threshold -> DROP quality
            - else -> KEEP quality, and store neighbors as context

        Tuning:
            Start around 0.50-0.65 depending on our embedding model and corpus.
            Lower thresholds keep more candidates; higher thresholds reduce LLM load.
    """
    encoder: TextEncoder
    top_k: int = 5
    threshold: float = 0.50
    console: Console = field(default_factory=Console) # console is a member so we can inject a test console or reuse a global one.

    # ------------------------------------------------------------------
    # Index building (KG side)
    # ------------------------------------------------------------------
    def build_index(self, relations: Sequence[GraphRelation]) -> VectorIndex[GraphRelation]:
        """
        Build a vector index over KG relations (subgraph only).

        Parameters
        ----------
        relations:
            The KG relations returned by the Graph Retriever module. This is
            typically a local subgraph around a keyword.

        Returns
        -------
        VectorIndex[GraphRelation]
            An in-memory vector index mapping vectors back to GraphRelation payloads.

        Raises
        ------
        ValueError:
            If `relations` is empty.

        Notes
        -----
        We intentionally index only the retrieved subgraph (not the entire KG) because:
        - it is smaller and faster
        - it matches the user's keyword context
        - it supports the "local deduplication" use case in our diagram
        """
        if not relations:
            raise ValueError("Cannot build vector index: 'relations' is empty.")

        # 1. Convert relations -> texts
        texts = [relation_to_text(r) for r in relations]

        # 2. Embed texts -> vectors
        vectors = self.encoder.encode(texts)

        # 3. Create an index sized exactly for this subgraph
        index = VectorIndex[GraphRelation].create(
            dim=self.encoder.dim,
            max_elements=len(relations),
        )

        # 4. Add vectors and preserve original GraphRelation objects as payloads
        index.add(vectors, relations)

        return index

    # ------------------------------------------------------------------
    # Filtering (quality side)
    # ------------------------------------------------------------------
    def filter_qualities(
        self,
        *,
        cfg: SubgraphSimilarityFilterConfig,
        index: VectorIndex[GraphRelation],
        qualities: Sequence[Quality],
        progress: Optional[ProgressCallback] = None
    ) -> Tuple[List[KeptQuality], List[DroppedQuality]]:
        """
        Filter candidate qualities using vector similarity to the KG index.

        Parameters
        ----------
        index:
            A VectorIndex built over KG relations (via build_index()).

        qualities:
            Atomic sentences produced by the Decomposer module.
            These are the *candidate claims* that we want to verify/dedupe.

        Returns
        -------
        (kept, dropped):
            kept:
                Each kept item contains:
                    - the quality text
                    - the maximum similarity score (similarity to nearest KG relation)
                    - the top-k nearest neighbors (KG relations) as context

            dropped:
                Each dropped item contains:
                    - the quality text
                    - the max similarity score (for debugging/tuning)

        Implementation details
        ----------------------
            - embeds all qualities in one batch (already done)
            - performs **one batch k-NN search** for all qualities
            - thresholding is vectorized
            - only lightweight Python work remains to assemble results
            - This removes the bottleneck of calling `index.search(...)` 1000+ times.

        This method does NOT:
            - call any LLM
            - extract triplets
            - write to Neo4j
        """
        total = 4
        step = 0
        def tick(msg: str):
            nonlocal step
            step += 1
            if progress:
                progress(step, total, msg)


        if not qualities:
            return ([], [])

        # 1. Convert qualities -> embedding texts (currently identity)
        texts = [quality_to_text(q) for q in qualities]

        # 2. Batch-embed qualities
        tick("🧬 Embedding qualities…")
        vectors = self.encoder.encode(texts) # shape: (num_qualities, dim)
        vectors_np = np.ascontiguousarray(np.asarray(vectors), dtype=np.float32)

        kept: List[KeptQuality] = []
        dropped: List[DroppedQuality] = []


        # 3) ❤️ Batch ANN (Approximate Nearest Neighbor) search: 
        #    returns (neighbors_per_quality, scores_matrix)
        #       - neighbors_per_quality: 
        #           - shape: (Q, k) 
        #               - where Q = number of qualities, 
        #               - k = top-k closest neighbors to the quality
        #               - i.e. List[List[GraphRelation]] length Q
        #       
        #       - scores_matrix: np.ndarray shape (Q, k) of cosine similarities
        with stage_status("📊 Performing batch vector similarity search:"):
            tick("📊 Searching nearest neighbors…")
            neighbors_per_q, scores = index.search_batch(vectors_np, k=self.top_k)


        # 4) Vectorized max score per quality
        #    scores is (Q, k). max_scores is (Q,)
        #  i.e we keep only the max score for each quality
        if scores.size == 0:
            max_scores = np.zeros((len(qualities),), dtype=np.float32)
        else:
            max_scores = scores.max(axis=1)


        # 5) Vectorized thresholding (boolean mask)
        keep_mask = max_scores >= float(self.threshold)


        # 6) Assemble outputs (still a loop, but *cheap*; no ANN calls inside)
        tick("🧮 Thresholding + assembling results…")
        for i, q in enumerate(qualities):
            ms = float(max_scores[i])

            if not keep_mask[i]:
                dropped.append({"quality": q, "max_score": ms})
                continue

            neighs = neighbors_per_q[i] # shape (k,)
            row_scores = scores[i]  # shape (k,)

            # Important: neighbors_per_q[i] may be shorter than k if labels contained -1.
            # We align by taking min length.
            n = min(len(neighs), int(row_scores.shape[0]))

            kept.append(
                {
                    "quality": q,
                    "max_score": ms,
                    "neighbors": [
                        {"relation": neighs[j], "score": float(row_scores[j])}
                        for j in range(n)
                    ],
                }
            )

        kept.sort(key=lambda x: x["max_score"], reverse=True)
        dropped.sort(key=lambda x: x["max_score"], reverse=True)

        tick("💾 Saving similarity results to JSON log…")
        self.save_similarity_results_json(cfg=cfg, kept=kept, dropped=dropped)
        return (kept, dropped)


    def pretty_print(
        self,
        *,
        kept: Sequence[KeptQuality],
        dropped: Sequence[DroppedQuality],
        title: str = "Vector Similarity Filter Results",
        max_neighbors_to_show: int = 3,
    ) -> None:
        """
        Pretty-print the results of vector similarity filtering using rich.

        Parameters
        ----------
        kept:
            Qualities that passed the similarity threshold.

        dropped:
            Qualities that were filtered out.

        title:
            Title shown at the top of the output.

        max_neighbors_to_show:
            Maximum number of nearest KG relations to display per kept quality.
            This affects only printing, not filtering logic.
        """
        self.console.rule(f"[bold cyan]{title}[/bold cyan]")

        # ----------------------------
        # Kept qualities
        # ----------------------------
        if kept:
            self.console.print(Rule("[bold green]🏆️✅ Kept Qualities[/bold green]"))

            for i, item in enumerate(kept, start=1):
                quality = item["quality"]
                max_score = item["max_score"]
                neighbors = item["neighbors"][:max_neighbors_to_show]

                header = Text()
                header.append(f"[{i}] ", style="bold green")
                header.append(f"(max_score={max_score:.3f})", style="dim")

                body_lines: list[str] = []
                body_lines.append(f"[bold]Quality:[/bold] {quality}")

                if neighbors:
                    body_lines.append("\n[bold]🧑‍🤝‍🧑 Nearest KG relations:[/bold]")
                    for j, nh in enumerate(neighbors, start=1):
                        rel = nh["relation"]
                        score = nh["score"]

                        props = rel["edge"]["properties"]
                        sentence = props.get("sentence")
                        source = props.get("source")
                        page = props.get("page_number")

                        line = (
                            f"  {j}. "
                            f"[magenta]{rel['source']['label']}[/magenta] "
                            f"— [white]{rel['edge']['label']}[/white] → "
                            f"[magenta]{rel['target']['label']}[/magenta] "
                            f"(score={score:.3f})"
                        )
                        body_lines.append(line)

                        if sentence:
                            body_lines.append(f"     sentence: {sentence}")

                        if source:
                            meta = f"{source}"
                            if page is not None:
                                meta += f", page {page}"
                            body_lines.append(f"     source: {meta}")

                self.console.print(
                    Panel(
                        "\n".join(body_lines),
                        title=header,
                        border_style="green",
                        padding=(1, 2),
                    )
                )

        else:
            self.console.print("[yellow]No qualities passed the similarity filter.[/yellow]")

        # ----------------------------
        # Dropped qualities
        # ----------------------------
        if dropped:
            self.console.print(Rule("[bold red]❌☹️ Dropped Qualities[/bold red]"))

            for i, item in enumerate(dropped, start=1):
                quality = item["quality"]
                max_score = item["max_score"]

                self.console.print(
                    Panel(
                        f"[bold]Quality:[/bold] {quality}\n"
                        f"[bold]Max similarity:[/bold] {max_score:.3f}",
                        title=f"[red]{i}[/red]",
                        border_style="red",
                        padding=(1, 2),
                    )
                )


    def save_similarity_results_json(
        self,
        *,
        cfg: SubgraphSimilarityFilterConfig,
        kept: Sequence[KeptQuality],
        dropped: Sequence[DroppedQuality],
    ) -> Mapping[str, Any]:
        """
        Write vector similarity filter results to a JSON file.

        The output structure is intentionally explicit and stable so it can be:
        - inspected manually
        - diffed across runs
        - reused by downstream experiments

        JSON structure:
        {
            "kept": [...],
            "dropped": [...]
        }
        """
        # created_at = now_utc_compact()
        # data: Mapping[str, Any] = {
        #     "number_kept": len(kept),
        #     "number_dropped": len(dropped),
        #     "kept": kept,
        #     "dropped": dropped,
        #     "created_at": created_at,
        # }
        log_payload = build_qualities_to_subgraph_similarity_payload(
            cfg=cfg,
            kept=kept,
            dropped=dropped,
        )

        path = f"logs/03_vector_similarity_filter_results_{now_utc_compact()}.json"
        write_json(path, log_payload)

        print(f"\n[INFO] 📚️📊 Wrote vector similarity results to {path}")

        return log_payload
    
