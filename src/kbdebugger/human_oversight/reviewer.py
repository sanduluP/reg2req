from __future__ import annotations

from typing import Iterable, List, Tuple

from kbdebugger.graph.utils import map_extracted_triplets_to_graph_relations
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
# from rich.rule import Rule
# from rich.text import Text

from kbdebugger.graph.store import GraphStore
from kbdebugger.types import GraphRelation
from kbdebugger.graph import get_graph
from kbdebugger.types import ExtractionResult
from .logger import save_human_oversight_log

console = Console()


def review_triplets(
    extraction_results: Iterable[ExtractionResult]
) -> Tuple[List[GraphRelation], List[GraphRelation]]:
    """
    Human-in-the-loop review for extracted triplets.

    For each relation:
    - Show it clearly using rich
    - Ask user to Accept / Reject / Quit
    - Accepted relations are upserted immediately

    Returns:
        (accepted_relations, rejected_relations)
    """
    graph = get_graph()

    relations: List[GraphRelation] = []
    for r in extraction_results:
        relations.extend(map_extracted_triplets_to_graph_relations(extraction=r))

    accepted: List[GraphRelation] = []
    rejected: List[GraphRelation] = []

    console.rule("[bold cyan]🧑‍⚖️ Human Oversight — Triplet Review[/bold cyan]")

    for i, rel in enumerate(relations, start=1):
        src = rel["source"]["label"]
        tgt = rel["target"]["label"]
        edge = rel["edge"]["label"]
        props = rel["edge"].get("properties", {})

        sentence = props.get("sentence", "")
        source = props.get("provenance_source") or props.get("source")
        page = props.get("page_number")

        body = []
        body.append(f"[bold]Relation:[/bold] {src} ── {edge} ──▶ {tgt}")
        if sentence:
            body.append(f"\n[bold]Sentence:[/bold] {sentence}")
        if source:
            meta = source
            if page is not None:
                meta += f", page {page}"
            body.append(f"\n[dim]Source:[/dim] {meta}")

        console.print(
            Panel(
                "\n".join(body),
                title=f"[bold]{i}[/bold]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        action = Prompt.ask(
            "[bold green]Accept[/bold green] (a) / "
            "[bold red]Reject[/bold red] (r) / "
            "[bold yellow]Quit[/bold yellow] (q)",
            choices=["a", "r", "q"],
            default="a",
        )

        if action == "a":
            graph.upsert_relation(rel)
            accepted.append(rel)
            console.print("[green]✔ Accepted and inserted into KG[/green]\n")

        elif action == "r":
            rejected.append(rel)
            console.print("[red]✖ Rejected[/red]\n")

        elif action == "q":
            console.print("[yellow]⏹ Review stopped by user[/yellow]")
            break
    
    # save_human_oversight_log(accepted=accepted, rejected=rejected)

    return accepted, rejected
