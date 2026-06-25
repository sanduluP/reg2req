"""
Phase B: five-strategy knowledge-graph verification.

A read-only verifier that checks the graph produced by the pipeline against the
source corpus through five complementary strategies (Coverage, Correctness,
Consistency, Completeness, Minimality). A run is PASS only if all five pass
their thresholds.

The verifier never mutates the graph — it only reads and reports — so it can be
run after the pipeline builds the graph without any risk to the data, and a
FAIL is purely a diagnostic signal.
"""

from .verifier import verify_graph, DEFAULT_THRESHOLDS, STRATEGY_NAMES

__all__ = ["verify_graph", "DEFAULT_THRESHOLDS", "STRATEGY_NAMES"]
