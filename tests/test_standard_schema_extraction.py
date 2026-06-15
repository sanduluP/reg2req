from __future__ import annotations

from pathlib import Path

from kbdebugger.standard_schema import (
    build_schema_grounding_context,
    build_schema_grounding_contexts,
    load_standard_schema_profile,
    validate_extraction_with_standard_schema,
)


def write_source(tmp_path: Path) -> Path:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "mini.py").write_text(
        'knowledge = """\n'
        'sklearn.metrics.accuracy_score is an Operator\n'
        'sklearn.metrics.accuracy_score implements Accuracy\n'
        'sklearn.metrics.accuracy_score evaluates Classification\n'
        'Accuracy is a Metric\n'
        'Accuracy contributes to Predictive Performance\n'
        'confusion_matrix has output matrix; is of type ndarray\n'
        'confusion_matrix is an Operator\n'
        'sklearn.model_selection.GridSearchCV is an Operator\n'
        'sklearn.model_selection.GridSearchCV performs Model Selection\n'
        'Model Selection is a Data Science Task\n'
        'sklearn.gaussian_process.kernels.Hyperparameter is a Parameter\n'
        'Class Imbalance is a Risk\n'
        'class_imbalance checks for Class Imbalance\n'
        '"""\n',
        encoding="utf-8",
    )
    return source_dir


def test_standard_schema_profile_extracts_nodes_types_templates_and_examples(tmp_path: Path) -> None:
    source_dir = write_source(tmp_path)
    profile = load_standard_schema_profile(str(source_dir))

    assert "sklearn.metrics.accuracy_score" in profile.canonical_nodes
    assert profile.node_types["sklearn.metrics.accuracy_score"] == ("Operator",)
    assert "Metric" in profile.node_class_names
    assert "Operator --Implements--> Metric" in profile.predicate_templates["Implements"]
    assert any("accuracy_score evaluates Classification" in line for line in profile.predicate_examples["Evaluates"])


def test_schema_grounding_finds_operator_metric_matrix_and_risk(tmp_path: Path) -> None:
    source_dir = write_source(tmp_path)
    profile = load_standard_schema_profile(str(source_dir))

    metric_ctx = build_schema_grounding_context(
        "Accuracy score evaluates classification quality.",
        profile,
    )
    assert metric_ctx.has_fit
    assert "Evaluates" in metric_ctx.predicate_hints
    assert any(match.node == "Accuracy" for match in metric_ctx.matched_nodes)

    matrix_ctx = build_schema_grounding_context(
        "The confusion matrix output is an ndarray matrix.",
        profile,
    )
    assert matrix_ctx.has_fit
    assert "DataType" in matrix_ctx.inferred_node_types
    assert "HasOutput" in matrix_ctx.predicate_hints

    risk_ctx = build_schema_grounding_context(
        "Class imbalance risk should be checked.",
        profile,
    )
    assert risk_ctx.has_fit
    assert any(match.node == "Class Imbalance" for match in risk_ctx.matched_nodes)
    assert "Risk" in risk_ctx.inferred_node_types


def test_schema_validation_marks_valid_and_generic_triplets(tmp_path: Path) -> None:
    source_dir = write_source(tmp_path)
    profile = load_standard_schema_profile(str(source_dir))

    contexts = build_schema_grounding_contexts(
        [
            "Accuracy score evaluates classification quality.",
            "Complex architectures make AI models difficult to trust.",
        ],
        source_path=str(source_dir),
    )

    valid = validate_extraction_with_standard_schema(
        {
            "sentence": "Accuracy score evaluates classification quality.",
            "triplets": [("accuracy_score", "Classification", "Evaluates")],
        },
        contexts[0],
        profile=profile,
    )
    assert valid["schema_status"] == "SCHEMA_VALID"
    assert valid["triplets"][0][0] == "accuracy_score"
    assert valid["matched_schema_nodes"]

    generic = validate_extraction_with_standard_schema(
        {
            "sentence": "Complex architectures make AI models difficult to trust.",
            "triplets": [("complex architectures", "AI models", "Ensures")],
        },
        contexts[1],
        profile=profile,
    )
    assert generic["schema_status"] in {"NEEDS_SCHEMA_REVIEW", "NO_SCHEMA_FIT"}
    assert generic["schema_notes"]


def test_schema_validation_does_not_canonicalize_grid_search_to_schema_node(tmp_path: Path) -> None:
    source_dir = write_source(tmp_path)
    profile = load_standard_schema_profile(str(source_dir))
    context = build_schema_grounding_context(
        "Grid search can enhance robustness by fine-tuning settings.",
        profile,
    )

    result = validate_extraction_with_standard_schema(
        {
            "sentence": "Grid search can enhance robustness by fine-tuning settings.",
            "triplets": [("grid", "Model Selection", "IsSubclassOf")],
        },
        context,
        profile=profile,
    )

    assert result["triplets"] == [("grid", "Model Selection", "IsSubclassOf")]
    assert result["schema_status"] == "NO_SCHEMA_FIT"
    assert any("Object 'Model Selection' is not supported" in note for note in result["schema_notes"])


def test_schema_validation_does_not_mark_valid_from_context_match_alone(tmp_path: Path) -> None:
    source_dir = write_source(tmp_path)
    profile = load_standard_schema_profile(str(source_dir))
    context = build_schema_grounding_context(
        "Accuracy score supports classification quality.",
        profile,
    )

    assert context.matched_nodes
    result = validate_extraction_with_standard_schema(
        {
            "sentence": "Accuracy score supports classification quality.",
            "triplets": [("Accuracy score", "classification quality", "Ensures")],
        },
        context,
        profile=profile,
    )

    assert result["triplets"] == [("Accuracy score", "classification quality", "Ensures")]
    assert result["schema_status"] != "SCHEMA_VALID"


def test_schema_validation_rejects_unsupported_sklearn_hyperparameter_path(tmp_path: Path) -> None:
    source_dir = write_source(tmp_path)
    profile = load_standard_schema_profile(str(source_dir))
    context = build_schema_grounding_context(
        "Hyperparameters are critical for robustness.",
        profile,
    )

    result = validate_extraction_with_standard_schema(
        {
            "sentence": "Hyperparameters are critical for robustness.",
            "triplets": [("sklearn.gaussian_process.kernels.Hyperparameter", "robustness", "IsA")],
        },
        context,
        profile=profile,
    )

    assert result["triplets"][0][0] == "sklearn.gaussian_process.kernels.Hyperparameter"
    assert result["schema_status"] == "NO_SCHEMA_FIT"
    assert any("Subject 'sklearn.gaussian_process.kernels.Hyperparameter' is not supported" in note for note in result["schema_notes"])


def test_schema_validation_keeps_non_standard_predicates_with_flag(tmp_path: Path) -> None:
    source_dir = write_source(tmp_path)
    profile = load_standard_schema_profile(str(source_dir))
    context = build_schema_grounding_context(
        "AI providers shall document model limitations.",
        profile,
    )

    result = validate_extraction_with_standard_schema(
        {
            "sentence": "AI providers shall document model limitations.",
            "triplets": [("AI provider", "model limitations", "MustDocument")],
        },
        context,
        profile=profile,
    )

    # Non-standard predicate is kept, not silently dropped.
    assert result["triplets"] == [("AI provider", "model limitations", "MustDocument")]
    assert result["non_standard_predicates"] == ["MustDocument"]

    statuses = [
        item["schema_status"]
        for item in result["schema_grounding"]["triplet_statuses"]
    ]
    assert "NON_STANDARD_PREDICATE" in statuses
    assert result["schema_status"] != "SCHEMA_VALID"
    assert any("not in the standard predicate list" in note for note in result["schema_notes"])


def test_schema_validation_accepts_source_faithful_schema_shaped_triplet(tmp_path: Path) -> None:
    source_dir = write_source(tmp_path)
    profile = load_standard_schema_profile(str(source_dir))
    context = build_schema_grounding_context(
        "The confusion matrix has output matrix.",
        profile,
    )

    result = validate_extraction_with_standard_schema(
        {
            "sentence": "The confusion matrix has output matrix.",
            "triplets": [("confusion matrix", "matrix", "HasOutput")],
        },
        context,
        profile=profile,
    )

    assert result["triplets"] == [("confusion matrix", "matrix", "HasOutput")]
    assert result["schema_status"] == "SCHEMA_VALID"
