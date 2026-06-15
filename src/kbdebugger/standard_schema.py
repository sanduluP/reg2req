from __future__ import annotations

import ast
import difflib
import re
import runpy
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from kbdebugger.extraction.predicate_options import DEFAULT_ALLOWED_PREDICATES
from kbdebugger.types import ExtractionResult, GraphRelation, TripletSubjectObjectPredicate


_CAMEL_BOUNDARY_RE = re.compile(r"(?<!^)(?=[A-Z])")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#.-]*")
_WORD_RE = re.compile(r"[a-z0-9+#.-]+")

_PREDICATE_TO_SNAKE: dict[str, str] = {
    predicate: _CAMEL_BOUNDARY_RE.sub("_", predicate).lower()
    for predicate in DEFAULT_ALLOWED_PREDICATES
}
_SNAKE_TO_PREDICATE: dict[str, str] = {
    snake: predicate
    for predicate, snake in _PREDICATE_TO_SNAKE.items()
}
_PREDICATE_PHRASES: list[tuple[str, str, re.Pattern[str]]] = [
    (
        snake,
        snake.replace("_", " "),
        re.compile(rf"(?<![A-Za-z0-9_]){re.escape(snake.replace('_', ' '))}(?![A-Za-z0-9_])"),
    )
    for snake in _PREDICATE_TO_SNAKE.values()
]

_ROLE_BY_PREDICATE: dict[str, str] = {
    "HasParameter": "Parameter",
    "WithParameter": "Parameter",
    "HasInput": "Input",
    "HasOutput": "Output",
    "HasAttribute": "Attribute",
    "HasPackage": "Package",
    "HasPosition": "Position",
    "HasName": "Name",
    "IsOfType": "DataType",
    "WithDefault": "Default",
    "WithChoices": "ChoiceSet",
    "WithLow": "LowerBound",
    "WithHigh": "UpperBound",
    "HasThreshold": "Threshold",
    "HasDecisionFunction": "DecisionFunction",
    "MightIntroduce": "Risk",
    "MightMitigate": "Risk",
    "IsThreatTo": "Principle",
    "ChecksFor": "Risk",
    "SurfacesRisk": "Risk",
    "SensitiveFamily": "ModelFamily",
    "BelongsToFamily": "ModelFamily",
    "Evaluates": "Data Science Task",
    "Performs": "Data Science Task",
    "AndPerformsTask": "Data Science Task",
    "SuggestsPreprocessing": "Operator",
    "SuggestsReplacement": "ModelFamily",
}

_WEAK_GENERIC_PREDICATES = {"Ensures", "ContributesTo", "Has", "Implements", "IsA", "IsAn"}

# Normative predicates from standards documents are schema-independent: they
# don't need a template in the grounding graph to be valid, only source support.
_NORMATIVE_PREDICATES = {"Requires", "Recommends", "Permits", "Prohibits", "Defines"}


@dataclass(frozen=True, slots=True)
class SchemaStatement:
    source_file: str
    subject: str
    predicate: str
    object: str
    raw: str
    modifiers: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class SchemaNodeMatch:
    query: str
    node: str
    score: float
    match_type: str
    node_types: tuple[str, ...] = ()


@dataclass(slots=True)
class StandardSchemaProfile:
    source_path: str
    statements: tuple[SchemaStatement, ...]
    canonical_nodes: tuple[str, ...]
    node_types: Mapping[str, tuple[str, ...]]
    node_class_names: tuple[str, ...]
    alias_to_node: Mapping[str, str]
    predicate_examples: Mapping[str, tuple[str, ...]]
    predicate_templates: Mapping[str, tuple[str, ...]]
    statements_by_node: Mapping[str, tuple[SchemaStatement, ...]]


@dataclass(slots=True)
class SchemaGroundingContext:
    quality: str
    extracted_entities: tuple[str, ...]
    matched_nodes: tuple[SchemaNodeMatch, ...]
    inferred_node_types: tuple[str, ...]
    predicate_hints: tuple[str, ...]
    schema_templates: tuple[str, ...]
    examples: tuple[str, ...]
    grounding_confidence: float

    @property
    def has_fit(self) -> bool:
        return bool(self.matched_nodes or self.inferred_node_types or self.predicate_hints)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "extracted_entities": list(self.extracted_entities),
            "inferred_node_types": list(self.inferred_node_types[:8]),
            "predicate_hints": list(self.predicate_hints[:8]),
            "schema_templates": list(self.schema_templates[:8]),
            "grounding_confidence": round(float(self.grounding_confidence), 3),
        }

    def to_result_dict(self) -> dict[str, Any]:
        return {
            "extracted_entities": list(self.extracted_entities),
            "matched_schema_nodes": [match.node for match in self.matched_nodes[:8]],
            "inferred_node_types": list(self.inferred_node_types[:8]),
            "predicate_hints": list(self.predicate_hints[:8]),
            "schema_templates": list(self.schema_templates[:8]),
            "grounding_confidence": round(float(self.grounding_confidence), 3),
        }


def normalize_key(text: str) -> str:
    words = _WORD_RE.findall(str(text or "").replace("_", " ").lower())
    return " ".join(words)


def humanize_predicate(predicate: str) -> str:
    return _PREDICATE_TO_SNAKE.get(predicate, predicate).replace("_", " ")


def _compact_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _split_camel(text: str) -> str:
    return _CAMEL_BOUNDARY_RE.sub(" ", text).replace("_", " ")


def _load_knowledge_text(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "knowledge" for target in node.targets):
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    except Exception:
        pass

    ns = runpy.run_path(str(path))
    return str(ns.get("knowledge") or "")


def _find_predicate(part: str) -> tuple[str, re.Match[str]] | None:
    matches: list[tuple[int, int, str, re.Match[str]]] = []
    for order, (snake, _phrase, pattern) in enumerate(_PREDICATE_PHRASES):
        match = pattern.search(part)
        if match:
            matches.append((match.start(), order, snake, match))
    if not matches:
        return None
    _start, _order, snake, match = min(matches, key=lambda item: (item[0], item[1]))
    return snake, match


def _parse_modifier(part: str) -> tuple[str, str] | None:
    found = _find_predicate(part)
    if found is None:
        return None
    snake, match = found
    obj = _compact_text(part[match.end() :]).strip('"')
    if not obj:
        return None
    return (_SNAKE_TO_PREDICATE.get(snake, snake), obj)


def _parse_statement(line: str, source_file: str) -> SchemaStatement | None:
    parts = [part.strip() for part in line.split(";") if part.strip()]
    if not parts:
        return None
    first = parts[0]
    found = _find_predicate(first)
    if found is None:
        return None

    snake, match = found
    subject = _compact_text(first[: match.start()])
    obj = _compact_text(first[match.end() :]).strip('"')
    if not subject or not obj:
        return None

    modifiers = tuple(
        modifier
        for modifier in (_parse_modifier(part) for part in parts[1:])
        if modifier is not None
    )
    return SchemaStatement(
        source_file=source_file,
        subject=subject,
        predicate=_SNAKE_TO_PREDICATE.get(snake, snake),
        object=obj,
        raw=line,
        modifiers=modifiers,
    )


def _aliases_for_node(node: str) -> set[str]:
    aliases = {
        normalize_key(node),
        normalize_key(node.replace(".", " ")),
        normalize_key(_split_camel(node)),
    }
    if "." in node:
        short = node.rsplit(".", 1)[-1]
        aliases.add(normalize_key(short))
        aliases.add(normalize_key(_split_camel(short)))
    if "_" in node:
        aliases.add(normalize_key(node.replace("_", " ")))
    return {alias for alias in aliases if len(alias) >= 2}


def _infer_object_role(predicate: str, obj: str, node_types: Mapping[str, set[str]]) -> str:
    if obj in node_types and node_types[obj]:
        return "/".join(sorted(node_types[obj])[:3])
    if predicate in {"IsA", "IsAn", "IsSubclassOf"}:
        return obj
    return _ROLE_BY_PREDICATE.get(predicate, "Concept")


def _infer_subject_role(subject: str, node_types: Mapping[str, set[str]]) -> str:
    if subject in node_types and node_types[subject]:
        return "/".join(sorted(node_types[subject])[:3])
    return "Concept"


def _empty_profile(source_path: str) -> StandardSchemaProfile:
    return StandardSchemaProfile(
        source_path=source_path,
        statements=(),
        canonical_nodes=(),
        node_types={},
        node_class_names=(),
        alias_to_node={},
        predicate_examples={},
        predicate_templates={},
        statements_by_node={},
    )


def _load_source_statements(source_path: str) -> tuple[SchemaStatement, ...]:
    resolved = Path(source_path).expanduser()
    if not resolved.exists() or not resolved.is_dir():
        return ()

    statements: list[SchemaStatement] = []
    for path in sorted(resolved.glob("*.py")):
        text = _load_knowledge_text(path)
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            statement = _parse_statement(line, path.name)
            if statement is not None:
                statements.append(statement)
    return tuple(statements)


def _statement_from_graph_row(row: Mapping[str, Any]) -> SchemaStatement | None:
    source = _compact_text(str(row.get("source") or ""))
    target = _compact_text(str(row.get("target") or ""))
    rel_type = _compact_text(str(row.get("predicate") or ""))
    if not source or not target or not rel_type:
        return None

    predicate = _SNAKE_TO_PREDICATE.get(rel_type)
    if predicate is None:
        return None

    props = row.get("props")
    if not isinstance(props, Mapping):
        props = {}
    raw = _compact_text(str(props.get("sentence") or ""))
    if not raw:
        raw = f"{source} {rel_type.replace('_', ' ')} {target}"

    return SchemaStatement(
        source_file="neo4j",
        subject=source,
        predicate=predicate,
        object=target,
        raw=raw,
    )


def _load_graph_statements() -> tuple[SchemaStatement, ...]:
    try:
        from kbdebugger.graph.store import GraphStore

        graph = GraphStore.connect(verbose=False)
        try:
            rows = graph.query(
                """
                MATCH (n)-[r]->(m)
                WHERE n.name IS NOT NULL AND m.name IS NOT NULL
                RETURN n.name AS source, type(r) AS predicate, m.name AS target, properties(r) AS props
                LIMIT 60000
                """
            )
        finally:
            graph.close()
    except Exception:
        return ()

    return tuple(
        statement
        for statement in (_statement_from_graph_row(row) for row in rows)
        if statement is not None
    )


def _build_profile(source_path: str, statements: Sequence[SchemaStatement]) -> StandardSchemaProfile:
    if not statements:
        return _empty_profile(source_path)

    nodes = sorted({value for st in statements for value in (st.subject, st.object) if value})
    node_types_mut: dict[str, set[str]] = defaultdict(set)
    subclass_parent: dict[str, set[str]] = defaultdict(set)

    for st in statements:
        if st.predicate in {"IsA", "IsAn"}:
            node_types_mut[st.subject].add(st.object)
        elif st.predicate == "IsSubclassOf":
            subclass_parent[st.subject].add(st.object)

    changed = True
    while changed:
        changed = False
        for child, parents in subclass_parent.items():
            for parent in parents:
                for typ in node_types_mut.get(parent, set()):
                    if typ not in node_types_mut[child]:
                        node_types_mut[child].add(typ)
                        changed = True

    class_names = sorted(
        {
            typ
            for values in node_types_mut.values()
            for typ in values
        }
        | {st.object for st in statements if st.predicate in {"IsA", "IsAn"}}
        | set(_ROLE_BY_PREDICATE.values())
    )

    alias_to_node: dict[str, str] = {}
    for node in nodes:
        for alias in _aliases_for_node(node):
            alias_to_node.setdefault(alias, node)

    examples_by_pred: dict[str, list[str]] = defaultdict(list)
    template_counter: dict[str, Counter[str]] = defaultdict(Counter)
    statements_by_node: dict[str, list[SchemaStatement]] = defaultdict(list)

    for st in statements:
        if len(examples_by_pred[st.predicate]) < 12:
            examples_by_pred[st.predicate].append(st.raw)
        subject_role = _infer_subject_role(st.subject, node_types_mut)
        object_role = _infer_object_role(st.predicate, st.object, node_types_mut)
        template_counter[st.predicate][f"{subject_role} --{st.predicate}--> {object_role}"] += 1
        statements_by_node[normalize_key(st.subject)].append(st)
        statements_by_node[normalize_key(st.object)].append(st)

    return StandardSchemaProfile(
        source_path=source_path,
        statements=tuple(statements),
        canonical_nodes=tuple(nodes),
        node_types={key: tuple(sorted(value)) for key, value in node_types_mut.items()},
        node_class_names=tuple(class_names),
        alias_to_node=alias_to_node,
        predicate_examples={key: tuple(value) for key, value in examples_by_pred.items()},
        predicate_templates={
            key: tuple(template for template, _count in counter.most_common(8))
            for key, counter in template_counter.items()
        },
        statements_by_node={key: tuple(value) for key, value in statements_by_node.items()},
    )


@lru_cache(maxsize=8)
def load_standard_schema_profile(source_path: str | None = None) -> StandardSchemaProfile:
    if source_path:
        resolved = str(Path(source_path).expanduser())
        return _build_profile(resolved, _load_source_statements(resolved))

    return _build_profile("neo4j", _load_graph_statements())


def extract_schema_entities(text: str, *, max_entities: int = 10) -> tuple[str, ...]:
    tokens = [m.group(0).replace("_", " ") for m in _TOKEN_RE.finditer(text or "")]
    chunks: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        key = normalize_key(token)
        if key in ENGLISH_STOP_WORDS:
            if current:
                chunks.append(current)
                current = []
            continue
        current.append(token)
    if current:
        chunks.append(current)

    out: list[str] = []
    seen: set[str] = set()

    def add(phrase: str) -> None:
        phrase = _compact_text(phrase).strip(" .,:;()[]{}\"'")
        key = normalize_key(phrase)
        if len(key) < 3 or key in seen or key in ENGLISH_STOP_WORDS:
            return
        seen.add(key)
        out.append(phrase)

    for chunk in chunks:
        if len(out) >= max_entities:
            break
        if 1 < len(chunk) <= 5:
            add(" ".join(chunk))
        for n in (3, 2, 1):
            if len(out) >= max_entities or len(chunk) < n:
                continue
            for i in range(len(chunk) - n + 1):
                add(" ".join(chunk[i : i + n]))
                if len(out) >= max_entities:
                    break

    return tuple(out[:max_entities])


def match_schema_nodes(
    text: str,
    profile: StandardSchemaProfile,
    *,
    max_matches: int = 5,
    min_score: float = 0.78,
) -> tuple[SchemaNodeMatch, ...]:
    if not profile.canonical_nodes:
        return ()

    queries = [text, *extract_schema_entities(text, max_entities=10)]
    matches: dict[str, SchemaNodeMatch] = {}
    alias_keys = list(profile.alias_to_node.keys())

    for query in queries:
        qkey = normalize_key(query)
        if len(qkey) < 3:
            continue

        exact = profile.alias_to_node.get(qkey)
        if exact:
            matches[exact] = SchemaNodeMatch(
                query=query,
                node=exact,
                score=1.0,
                match_type="alias",
                node_types=profile.node_types.get(exact, ()),
            )
            continue

        if len(qkey) >= 5:
            for alias, node in profile.alias_to_node.items():
                if len(alias) < 5:
                    continue
                if f" {alias} " in f" {qkey} " or f" {qkey} " in f" {alias} ":
                    score = min(0.96, 0.72 + min(len(qkey), len(alias)) / max(len(qkey), len(alias), 1) * 0.24)
                    current = matches.get(node)
                    if current is None or score > current.score:
                        matches[node] = SchemaNodeMatch(
                            query=query,
                            node=node,
                            score=score,
                            match_type="substring",
                            node_types=profile.node_types.get(node, ()),
                        )

        close = difflib.get_close_matches(qkey, alias_keys, n=3, cutoff=min_score)
        for alias in close:
            node = profile.alias_to_node[alias]
            score = difflib.SequenceMatcher(None, qkey, alias).ratio()
            current = matches.get(node)
            if current is None or score > current.score:
                matches[node] = SchemaNodeMatch(
                    query=query,
                    node=node,
                    score=score,
                    match_type="fuzzy",
                    node_types=profile.node_types.get(node, ()),
                )

    return tuple(sorted(matches.values(), key=lambda match: match.score, reverse=True)[:max_matches])


def _infer_predicate_hints(text: str) -> tuple[str, ...]:
    key = normalize_key(text)
    hints: list[str] = []

    patterns: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("parameter", ("HasParameter",)),
        ("hyperparameter", ("HasParameter",)),
        ("input", ("HasInput",)),
        ("output", ("HasOutput",)),
        ("return", ("HasOutput",)),
        ("attribute", ("HasAttribute",)),
        ("metric", ("IsA", "Evaluates")),
        ("score", ("IsA", "Evaluates")),
        ("evaluate", ("Evaluates",)),
        ("evaluation", ("Evaluates", "IsA")),
        ("threshold", ("HasThreshold",)),
        ("risk", ("MightIntroduce", "SurfacesRisk")),
        ("bias", ("MightIntroduce", "IsThreatTo")),
        ("mitigate", ("MightMitigate",)),
        ("threat", ("IsThreatTo",)),
        ("family", ("BelongsToFamily", "SensitiveFamily")),
        ("model", ("IsA", "Performs", "Implements")),
        ("algorithm", ("Implements", "IsA")),
        ("operator", ("IsAn",)),
        ("function", ("IsAn", "Implements")),
        ("interface", ("IsA", "AppliesTo")),
        ("matrix", ("HasOutput", "IsOfType")),
        ("array", ("HasOutput", "IsOfType")),
        ("ndarray", ("HasOutput", "IsOfType")),
        ("tensor", ("HasOutput", "IsOfType")),
    )

    for needle, predicates in patterns:
        if needle in key:
            for predicate in predicates:
                if predicate not in hints:
                    hints.append(predicate)
    return tuple(hints)


def _infer_class_hints(text: str, profile: StandardSchemaProfile) -> tuple[str, ...]:
    key = normalize_key(text)
    hints: list[str] = []
    for class_name in profile.node_class_names:
        class_key = normalize_key(class_name)
        if len(class_key) >= 4 and f" {class_key} " in f" {key} ":
            hints.append(class_name)

    extra = {
        "matrix": "DataType",
        "array": "DataType",
        "ndarray": "DataType",
        "tensor": "DataType",
        "algorithm": "Concept",
        "evaluation": "Evaluation Procedure",
    }
    for needle, class_name in extra.items():
        if needle in key and class_name not in hints:
            hints.append(class_name)
    return tuple(hints[:8])


def build_schema_grounding_context(
    quality: str,
    profile: StandardSchemaProfile | None = None,
    *,
    max_matches: int = 5,
    max_examples: int = 8,
) -> SchemaGroundingContext:
    profile = profile or load_standard_schema_profile()
    entities = extract_schema_entities(quality, max_entities=10)
    matches = match_schema_nodes(quality, profile, max_matches=max_matches)
    class_hints = _infer_class_hints(quality, profile)
    predicate_hints = _infer_predicate_hints(quality)

    example_lines: list[str] = []
    template_lines: list[str] = []
    seen_examples: set[str] = set()
    seen_templates: set[str] = set()

    for match in matches:
        for st in profile.statements_by_node.get(normalize_key(match.node), ())[:4]:
            if st.raw not in seen_examples:
                seen_examples.add(st.raw)
                example_lines.append(st.raw)
            for template in profile.predicate_templates.get(st.predicate, ())[:2]:
                if template not in seen_templates:
                    seen_templates.add(template)
                    template_lines.append(template)

    for predicate in predicate_hints:
        for template in profile.predicate_templates.get(predicate, ())[:3]:
            if template not in seen_templates:
                seen_templates.add(template)
                template_lines.append(template)
        for example in profile.predicate_examples.get(predicate, ())[:2]:
            if example not in seen_examples:
                seen_examples.add(example)
                example_lines.append(example)

    if not template_lines:
        for predicate in ("IsAn", "IsA", "Implements", "HasParameter", "HasInput", "HasOutput", "IsOfType"):
            for template in profile.predicate_templates.get(predicate, ())[:1]:
                if template not in seen_templates:
                    seen_templates.add(template)
                    template_lines.append(template)

    confidence = max((match.score for match in matches), default=0.0)
    if class_hints and confidence < 0.72:
        confidence = 0.72
    if predicate_hints and confidence < 0.64:
        confidence = 0.64

    return SchemaGroundingContext(
        quality=quality,
        extracted_entities=entities,
        matched_nodes=matches,
        inferred_node_types=class_hints,
        predicate_hints=predicate_hints,
        schema_templates=tuple(template_lines[:max_examples]),
        examples=tuple(example_lines[:max_examples]),
        grounding_confidence=confidence,
    )


def build_schema_grounding_contexts(
    qualities: Sequence[str],
    *,
    source_path: str | None = None,
) -> list[SchemaGroundingContext]:
    profile = load_standard_schema_profile(source_path)
    return [build_schema_grounding_context(quality, profile) for quality in qualities]


def _singular_plural_variants(token: str) -> set[str]:
    variants = {token}
    if len(token) <= 3:
        return variants
    if token.endswith("ies"):
        variants.add(f"{token[:-3]}y")
    elif token.endswith("s"):
        variants.add(token[:-1])
    else:
        variants.add(f"{token}s")
    return variants


def _quality_supports_phrase(phrase: str, quality: str) -> bool:
    phrase_key = normalize_key(_split_camel(phrase))
    quality_key = normalize_key(_split_camel(quality))
    if not phrase_key or not quality_key:
        return False
    if f" {phrase_key} " in f" {quality_key} ":
        return True

    phrase_tokens = [token for token in phrase_key.split() if token not in ENGLISH_STOP_WORDS]
    if not phrase_tokens:
        return False

    quality_tokens = set(quality_key.split())
    return all(_singular_plural_variants(token) & quality_tokens for token in phrase_tokens)


def _object_can_be_schema_type(predicate: str, obj: str, profile: StandardSchemaProfile) -> bool:
    obj_key = normalize_key(_split_camel(obj))
    if not obj_key:
        return False

    class_keys = {normalize_key(_split_camel(name)) for name in profile.node_class_names}
    role_keys = {normalize_key(_split_camel(role)) for role in _ROLE_BY_PREDICATE.values()}
    if predicate in {"IsA", "IsAn"} and obj_key in class_keys | role_keys:
        return True
    if predicate == "IsOfType":
        return any(
            token in obj_key
            for token in ("str", "int", "float", "bool", "array", "ndarray", "tensor", "matrix", "dataframe", "dict", "list")
        )
    return False


def _object_fits_predicate(predicate: str, obj: str, profile: StandardSchemaProfile) -> bool:
    obj_key = normalize_key(obj)
    if not obj_key:
        return False
    if obj in profile.node_class_names or obj in profile.canonical_nodes:
        return True
    role = _ROLE_BY_PREDICATE.get(predicate)
    if role and normalize_key(role) in obj_key:
        return True
    if predicate in {"HasParameter", "HasInput", "HasOutput", "HasAttribute"}:
        return len(obj_key) >= 2
    if predicate == "IsOfType":
        return any(token in obj_key for token in ("str", "int", "float", "bool", "array", "ndarray", "tensor", "matrix", "dataframe", "dict", "list"))
    if predicate in {"MightIntroduce", "MightMitigate", "IsThreatTo", "ChecksFor", "SurfacesRisk"}:
        return any(token in obj_key for token in ("risk", "bias", "shift", "leakage", "noise", "privacy", "toxicity", "hallucination", "injection", "fairness", "robustness", "accountability"))
    return False


def validate_extraction_with_standard_schema(
    result: ExtractionResult,
    context: SchemaGroundingContext,
    *,
    profile: StandardSchemaProfile | None = None,
    allowed_predicates: Sequence[str] | None = None,
) -> ExtractionResult:
    profile = profile or load_standard_schema_profile()
    allowed = set(allowed_predicates or DEFAULT_ALLOWED_PREDICATES)
    cleaned_triplets: list[TripletSubjectObjectPredicate] = []
    triplet_statuses: list[dict[str, Any]] = []
    notes: list[str] = []
    non_standard_predicates: list[str] = []

    for subj, obj, predicate in result.get("triplets", []):
        if predicate not in allowed:
            # Keep the triplet so the reviewer can see and decide on it; just
            # flag the predicate as outside the standard predicate list.
            clean_subj = _compact_text(subj)
            clean_obj = _compact_text(obj)
            non_standard_predicates.append(predicate)
            notes.append(f"Predicate '{predicate}' is not in the standard predicate list.")
            cleaned_triplets.append((clean_subj, clean_obj, predicate))
            triplet_statuses.append(
                {
                    "subject": clean_subj,
                    "predicate": predicate,
                    "object": clean_obj,
                    "schema_status": "NON_STANDARD_PREDICATE",
                    "schema_template": f"Concept --{predicate}--> Concept",
                    "subject_match": None,
                    "object_match": None,
                    "subject_supported_by_quality": None,
                    "object_supported_by_quality": None,
                    "object_is_schema_type": None,
                }
            )
            continue

        clean_subj = _compact_text(subj)
        clean_obj = _compact_text(obj)
        quality_text = str(result.get("sentence") or context.quality or "")
        template = next(iter(profile.predicate_templates.get(predicate, ())), f"Concept --{predicate}--> Concept")

        subject_supported = _quality_supports_phrase(clean_subj, quality_text)
        object_supported = _quality_supports_phrase(clean_obj, quality_text)
        object_schema_type = _object_can_be_schema_type(predicate, clean_obj, profile)
        object_fit = _object_fits_predicate(predicate, clean_obj, profile)
        predicate_has_schema = predicate in profile.predicate_templates
        predicate_hint = predicate in context.predicate_hints

        if not subject_supported:
            status = "NO_SCHEMA_FIT"
            notes.append(f"Subject '{clean_subj}' is not supported by the source quality.")
        elif not object_supported and object_schema_type:
            status = "NEEDS_SCHEMA_REVIEW"
            notes.append(f"Object '{clean_obj}' is an inferred schema type not directly stated in the source quality.")
        elif not object_supported:
            status = "NO_SCHEMA_FIT"
            notes.append(f"Object '{clean_obj}' is not supported by the source quality.")
        elif predicate in _NORMATIVE_PREDICATES:
            # Source-supported normative assertions are valid regardless of the
            # grounding graph — standards text is not expected to fit ML-operator
            # templates, and tagging it "no fit" would just be noise.
            status = "SCHEMA_VALID"
        elif predicate in _WEAK_GENERIC_PREDICATES and not (object_fit or object_schema_type or predicate_hint):
            status = "NEEDS_SCHEMA_REVIEW"
            notes.append(f"{predicate} triplet is generic and should be reviewed against the standard schema.")
        elif predicate_has_schema and (object_fit or object_schema_type or predicate_hint):
            status = "SCHEMA_VALID"
        elif predicate_has_schema:
            status = "NEEDS_SCHEMA_REVIEW"
            notes.append(f"{predicate} exists in the standard schema but entity typing is uncertain.")
        else:
            status = "NO_SCHEMA_FIT"
            notes.append(f"{predicate} has no standard schema template.")

        cleaned_triplets.append((clean_subj, clean_obj, predicate))
        triplet_statuses.append(
            {
                "subject": clean_subj,
                "predicate": predicate,
                "object": clean_obj,
                "schema_status": status,
                "schema_template": template,
                "subject_match": None,
                "object_match": None,
                "subject_supported_by_quality": subject_supported,
                "object_supported_by_quality": object_supported,
                "object_is_schema_type": object_schema_type,
            }
        )

    result["triplets"] = cleaned_triplets

    statuses = {item["schema_status"] for item in triplet_statuses}
    if not cleaned_triplets:
        schema_status = "NO_SCHEMA_FIT" if not context.has_fit else "NEEDS_SCHEMA_REVIEW"
    elif statuses == {"SCHEMA_VALID"}:
        schema_status = "SCHEMA_VALID"
    elif "NO_SCHEMA_FIT" in statuses and len(statuses) == 1:
        schema_status = "NO_SCHEMA_FIT"
    else:
        schema_status = "NEEDS_SCHEMA_REVIEW"

    grounding = context.to_result_dict()
    grounding["triplet_statuses"] = triplet_statuses
    grounding["schema_notes"] = sorted(set(notes))[:8]

    if non_standard_predicates:
        result["non_standard_predicates"] = sorted(set(non_standard_predicates))  # type: ignore[typeddict-unknown-key]

    result["schema_status"] = schema_status  # type: ignore[typeddict-unknown-key]
    result["schema_template"] = "; ".join(dict.fromkeys(item["schema_template"] for item in triplet_statuses))  # type: ignore[typeddict-unknown-key]
    result["grounding_confidence"] = grounding["grounding_confidence"]  # type: ignore[typeddict-unknown-key]
    result["matched_schema_nodes"] = grounding["matched_schema_nodes"]  # type: ignore[typeddict-unknown-key]
    result["inferred_node_types"] = grounding["inferred_node_types"]  # type: ignore[typeddict-unknown-key]
    result["schema_notes"] = grounding["schema_notes"]  # type: ignore[typeddict-unknown-key]
    result["schema_grounding"] = grounding  # type: ignore[typeddict-unknown-key]
    return result


def graph_relation_from_statement(statement: SchemaStatement) -> GraphRelation:
    return {
        "source": {"label": statement.subject, "id": None, "created_at": None, "last_updated_at": None},
        "target": {"label": statement.object, "id": None, "created_at": None, "last_updated_at": None},
        "edge": {
            "label": _PREDICATE_TO_SNAKE.get(statement.predicate, statement.predicate),
            "properties": {"sentence": statement.raw, "source": f"schema:{statement.source_file}"},
        },
    }
