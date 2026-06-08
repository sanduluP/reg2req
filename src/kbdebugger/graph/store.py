from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence, cast
from typing_extensions import LiteralString
from dotenv import load_dotenv
from rich.progress import track

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError

from kbdebugger.types import GraphRelation
from .utils import predicate_to_dorian_relationship_type, rows_to_graph_relations
from .types import BatchUpsertSummary
from .aura_api import ensure_aura_running_from_env 

import rich
from rich.console import Console
from rich.panel import Panel

# Load env vars once here
load_dotenv(override=True)

@dataclass
class GraphStore:
    """
    Central access point to the knowledge graph.

    - Handles connecting to Neo4j
    - Exposes:
        - `query(...)` for arbitrary Cypher
        - `upsert_relation(...)` for writing extracted relations
    """
    # inner: Neo4jGraph
    driver: Driver

    # ---------- construction / connection ----------
    @classmethod
    def connect(
        cls,
        *,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        auto_env: bool = True,
        verbose: bool = True,
    ) -> "GraphStore":
        """
        Build a GraphStore from environment variables (or explicit args).

        Env vars:
            - NEO4J_URI
            - NEO4J_USERNAME (default "neo4j")
            - NEO4J_PASSWORD (default "")
        """
        if auto_env:
            load_dotenv(override=True)

        # ✅ Preflight: make sure Aura is running before Neo4j driver even tries DNS
        ensure_aura_running_from_env(verbose=verbose)

        neo4j_uri = uri or os.getenv("NEO4J_URI")
        neo4j_user = username or os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_pass = password or os.getenv("NEO4J_PASSWORD", "")

        if not neo4j_uri:
            raise RuntimeError("NEO4J_URI is not set (pass uri=... or set env var).")

        # inner = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_pass)
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))

        if verbose:
            rich.print(
                f"[kbdebugger] Connected to Neo4j at {neo4j_uri!r} "
                f"as user {neo4j_user!r}"
            )

        # return cls(inner=inner)
        return cls(driver=driver)


    def close(self) -> None:
        self.driver.close()


    # ---------- basic query API ----------
    def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run a Cypher query with consistent error handling.

        This is the *only* low-level escape hatch other code should use.
        """
        # print("GraphStore backend:", type(self.driver), type(self))
        try:
            with self.driver.session() as session:
                cypher_as_literal_str = cast(LiteralString, cypher)
                result = session.run(cypher_as_literal_str, params or {})
                return [record.data() for record in result]
        except Neo4jError as e:
            message = "Neo4j query failed!\n" \
                f"Error: {e.__class__.__name__}: {e}\n" \
                f"Query:\n{cypher}\n" \
                f"Params:\n{params}"
            
            rich.print(
                f"[bold red]{message}[/bold red]\n"
            )
            raise RuntimeError(message) from e
        except Exception as e:
            raise RuntimeError(
                "Unexpected error during Neo4j query:\n"
                f"{e}\nQuery:\n{cypher}\nParams:\n{params}"
            ) from e


    def query_relations(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        *,
        source_key: str = "source",
        target_key: str = "target",
        predicate_key: str = "predicate",
        props_key: str = "props",
    ) -> list[GraphRelation]:
        """
        Run a Cypher query that returns (source, target, predicate, props) columns
        and coerce it to List[GraphRelation].
        """
        rows = self.query(cypher, params=params or {})
        return rows_to_graph_relations(
            rows,
            source_key=source_key,
            target_key=target_key,
            predicate_key=predicate_key,
            props_key=props_key,
        )


    # ---------- high-level write API ----------
    def upsert_relation(self, relation: GraphRelation) -> list[dict[str, Any]]:
        """
        Insert or update a single GraphRelation using Dorian-style Neo4j.

        - Nodes are stored as `(:Node {name: ...})`
        - Predicates are stored as real relationship types, e.g. `:has_parameter`
        - Provenance remains on the relationship as properties
        - Dedupe is based on source node, target node, relationship type, and
          the provenance `source` property when present
        """
        src_name = str(relation["source"]["label"]).strip()
        tgt_name = str(relation["target"]["label"]).strip()
        rel_label = str(relation["edge"]["label"]).strip()

        if not src_name or not tgt_name:
            raise ValueError("GraphRelation source and target labels must be non-empty.")

        rel_type = predicate_to_dorian_relationship_type(rel_label)

        raw_props = relation["edge"].get("properties") or {}
        props_all = {
            str(k): v
            for k, v in raw_props.items()
            if v is not None and str(k) not in {"label", "type"}
        }

        now_iso = datetime.now(timezone.utc).isoformat()
        rel_source_key = str(props_all.get("source", ""))

        on_create_props = {
            **props_all,
            "source": rel_source_key,
            "created_at": now_iso,
            "last_updated_at": now_iso,
        }
        on_match_props = {
            **props_all,
            "source": rel_source_key,
            "last_updated_at": now_iso,
        }

        cypher = f"""
        MERGE (s:Node {{name: $source_name}})
        ON CREATE SET s.created_at = datetime()
        SET s.last_updated_at = datetime(),
            s.created_at = coalesce(s.created_at, datetime())

        MERGE (t:Node {{name: $target_name}})
        ON CREATE SET t.created_at = datetime()
        SET t.last_updated_at = datetime(),
            t.created_at = coalesce(t.created_at, datetime())

        MERGE (s)-[rel:`{rel_type}` {{source: $rel_source_key}}]->(t)
        ON CREATE SET rel += $on_create
        ON MATCH SET rel += $on_match

        RETURN s, t, rel
        """

        return self.query(
            cypher,
            params={
                "source_name": src_name,
                "target_name": tgt_name,
                "rel_source_key": rel_source_key,
                "on_create": on_create_props,
                "on_match": on_match_props,
            },
        )


    def upsert_relations(
            self, 
            relations: Sequence[GraphRelation],
            *,
            pretty_print: bool = True,
    ) -> BatchUpsertSummary:
        """
        Upsert multiple GraphRelation objects into Neo4j.

        This is a convenience wrapper around `upsert_relation()` that:
        - preserves the dedupe semantics of the single-upsert operation
        - continues on individual failures (best-effort write)
        - returns a typed summary for logging and monitoring

        Parameters
        ----------
        relations:
            Relations to be inserted/merged into the KG.

        Returns
        -------
        BatchUpsertSummary
            Counts and error strings describing any failures.

        Notes
        -----
        - This method performs no client-side deduplication.
          Deduplication is handled inside `upsert_relation()` via Dorian-style
          typed relationship MERGE semantics.
        - If you later want "fail-fast" semantics, add a flag like `stop_on_error`.
        """
        if not relations:
            # Nothing to upsert is not an error here; extraction may legitimately produce nothing.
            return BatchUpsertSummary(
                attempted=0,
                succeeded=0,
                failed=0,
                errors=[],
            )

        attempted = len(relations)
        succeeded = 0
        errors: List[str] = []

        for i, rel in track(
            enumerate(relations, start=1), 
            description="➕🛸 Upserting triplets (relations) into Knowledge Graph",
            total=len(relations)):
            try:
                self.upsert_relation(rel)
                succeeded += 1
            except Exception as e:  # pylint: disable=broad-exception-caught
                src = rel.get("source", {}).get("label", "?")
                tgt = rel.get("target", {}).get("label", "?")
                pred = rel.get("edge", {}).get("label", "?")
                errors.append(f"[{i}/{attempted}] {src} - {pred} -> {tgt}: {e}")

        failed = attempted - succeeded
        
        summary = BatchUpsertSummary(
            attempted=attempted,
            succeeded=succeeded,
            failed=failed,
            errors=errors,
        )

        if pretty_print:
            console = Console()

            body_lines = [
                f"[bold]Attempted:[/bold] {summary.attempted}",
                f"[bold green]Succeeded:[/bold green] {summary.succeeded}",
                f"[bold red]Failed:[/bold red] {summary.failed}",
            ]

            if summary.failed > 0:
                body_lines.append("\n[bold red]Errors:[/bold red]")
                for err in summary.errors:
                    body_lines.append(f"  • {err}")

            console.print(
                Panel(
                    "\n".join(body_lines),
                    title="[bold cyan]🧠📊 Knowledge Graph Upsert Summary[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )


        return summary
