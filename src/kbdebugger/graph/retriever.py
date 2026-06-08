from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence, TypedDict

import rich
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from kbdebugger.graph import get_graph
from kbdebugger.types import GraphRelation, EdgePropertyKey
from kbdebugger.utils.json import write_json
from kbdebugger.utils.time import now_utc_compact
from .utils import normalize_text

MatchPattern = Literal["source_name", "target_name", "rel_props"]

class RetrievedRelation(TypedDict):
    relation: GraphRelation
    match_pattern: MatchPattern

@dataclass
class KnowledgeGraphRetriever:
    """
    Keyword-guided KG retrieval.

    MVP: returns 1-hop 'path fragments' (edges) as normalized GraphRelation objects.
    """
    limit_per_pattern: int = 50
    console: Console = field(default_factory=Console) # console is a member so we can inject a test console or reuse a global one.

    def retrieve(
        self,
        keyword: str,
        *,
        limit_per_pattern: Optional[int] = None,
    ) -> list[RetrievedRelation]:
        
        kw = normalize_text(keyword)
        limit = int(limit_per_pattern or self.limit_per_pattern)

        graph = get_graph()

        results: list[RetrievedRelation] = []

        # --- Pattern 1: keyword in source node name ---
        rels = graph.query_relations(
            """
            MATCH (n:Node)-[r]->(m:Node)
            WHERE toLower(n.name) CONTAINS $keyword
            RETURN
              n.name AS source,
              m.name AS target,
              type(r) AS predicate,
              properties(r) AS props,

              elementId(n) AS source_id,
              elementId(m) AS target_id,
              elementId(r) AS rel_id
            LIMIT $limit
            """,
            {"keyword": kw, "limit": limit},
        )
        results.extend({"relation": rel, "match_pattern": "source_name"} for rel in rels)


        # --- Pattern 2: keyword in target node name ---
        rels = graph.query_relations(
            """
            MATCH (n:Node)-[r]->(m:Node)
            WHERE toLower(m.name) CONTAINS $keyword
            RETURN
              n.name AS source,
              m.name AS target,
              type(r) AS predicate,
              properties(r) AS props,

              elementId(n) AS source_id,
              elementId(m) AS target_id,
              elementId(r) AS rel_id
            LIMIT $limit
            """,
            {"keyword": kw, "limit": limit},
        )
        results.extend({"relation": rel, "match_pattern": "target_name"} for rel in rels)

        # --- Pattern 3: keyword in Dorian relationship type or provenance fields ---
        rels = graph.query_relations(
            """
            MATCH (n:Node)-[r]->(m:Node)
            WHERE
                toLower(type(r))                            CONTAINS $keyword OR
                toLower(replace(type(r), "_", " "))        CONTAINS $keyword OR
                toLower(coalesce(r.sentence, ""))          CONTAINS $keyword OR
                toLower(coalesce(r.source, ""))            CONTAINS $keyword
            RETURN
                n.name AS source,
                m.name AS target,
                type(r) AS predicate,
                properties(r) AS props,

                elementId(n) AS source_id,
                elementId(m) AS target_id,
                elementId(r) AS rel_id
            LIMIT $limit
            """,
            {"keyword": kw, "limit": limit},
        )
        results.extend({"relation": rel, "match_pattern": "rel_props"} for rel in rels)

        # Optional: dedupe identical relations across patterns
        # (same source/target/predicate + same sentence/source if you want)
        results = self._dedupe(results)

        # # TODO: Enable again after we're done with Frontend integration
        # self.save_results_json(
        #     keyword=keyword,
        #     hits=results,
        #     limit_per_pattern=limit,
        # )

        return results


    @staticmethod
    def _dedupe(items: list[RetrievedRelation]) -> list[RetrievedRelation]:
        seen: set[tuple[str, str, str, str]] = set()
        out: list[RetrievedRelation] = []

        for item in items:
            rel = item["relation"]
            props = rel["edge"]["properties"]
            sentence = str(props.get("sentence", ""))  # good lightweight key
            key = (rel["source"]["label"], rel["target"]["label"], rel["edge"]["label"], sentence)

            if key in seen:
                continue
            seen.add(key)
            out.append(item)

        return out


    @staticmethod
    def save_results_json(
        *,
        keyword: str,
        hits: Sequence[RetrievedRelation],
        limit_per_pattern: int | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Save KG retrieval results to a JSON log file.

        Parameters
        ----------
        keyword:
            The user keyword used to retrieve the subgraph.

        hits:
            The retrieved results (a list of RetrievedRelation).
            Each element contains:
                - relation: GraphRelation
                - match_pattern: str (provenance of how it matched)

        limit_per_pattern:
            Optional: include the retriever's `limit_per_pattern` setting in the log.
            i.e., how many relations were retrieved per `MatchPattern`.

        extra_metadata:
            Optional: additional metadata to include in the JSON file
            (e.g., commit hash, run id, corpus file name, etc.).

        JSON Schema (high-level)
        ------------------------
        {
          "keyword": "...",
          "limit_per_pattern": 50,
          "num_hits": 123,
          "hits": [...],
          "extra": {...}
        }
        """
        created_at = now_utc_compact()
        payload: dict[str, Any] = {
            "keyword": keyword,
            "limit_per_pattern": limit_per_pattern,
            "num_hits": len(hits),
            "hits": list(hits),
            "created_at": created_at,
        }

        if extra_metadata:
            payload["extra"] = dict(extra_metadata)

        path = f"logs/02_kg_retrieval_{keyword}_{created_at}.json"
        write_json(path, payload)

        rich.print(f"\n[INFO] Wrote KG retrieval log to {path}")


    def pretty_print(
        self,
        hits: Sequence[RetrievedRelation],
        *,
        title: str = "Knowledge Graph Retrieval Results",
        show_props_keys: Optional[Sequence[EdgePropertyKey]] = None,
    ) -> None:
        """
        Pretty-print RetrievedRelation results using rich.

        - hits: output of retrieve(..., include_match_pattern=True)
        - show_props_keys: if set, prints only these keys from edge.properties (in addition to sentence/source/page)
        """
        if not hits:
            self.console.rich.print("[bold yellow]No matching relations found.[/bold yellow]")
            return

        if not show_props_keys:
            show_props_keys = [
                "created_at", 
                "last_updated_at",
                "original_sentence",
            ]

        self.console.rule(f"[bold cyan]{title}[/bold cyan]")

        for i, hit in enumerate(hits, start=1):
            rel = hit["relation"]
            pattern = hit["match_pattern"]

            src = rel["source"]["label"]
            tgt = rel["target"]["label"]
            pred = rel["edge"]["label"]
            props = rel["edge"]["properties"]

            sentence = props.get("sentence")
            source_doc = props.get("source")
            page = props.get("page_number")

            header = Text()
            header.append(f"[{i}] ", style="bold cyan")
            header.append(f"{src} ", style="bold green")
            header.append("── ", style="dim")
            header.append(pred, style="bold magenta")
            header.append(" ──> ", style="dim")
            header.append(tgt, style="bold green")

            body: list[str] = [f"[bold]Matched via:[/bold] {pattern}"]

            if sentence:
                body.append(f"[bold]Sentence:[/bold] {sentence}")

            if source_doc:
                meta = f"{source_doc}"
                if page is not None:
                    meta += f", page {page}"
                body.append(f"[bold]Source:[/bold] {meta}")

            if show_props_keys:
                for k in show_props_keys:
                    if k in props and props[k] is not None: # type: ignore
                        body.append(f"[bold]{k}:[/bold] {props[k]}") # type: ignore

            self.console.rich.print(
                Panel(
                    "\n".join(body),
                    title=header,
                    border_style="cyan",
                    padding=(1, 2),
                )
            )

