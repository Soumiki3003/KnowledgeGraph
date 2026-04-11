from app import models
from app.services.knowledge import _slim_knowledge_for_prompt


# ---------------------------------------------------------------------------
# RootKnowledge
# ---------------------------------------------------------------------------


def test_slim_root_type(root_knowledge):
    result = _slim_knowledge_for_prompt(root_knowledge)
    assert result["type"] == "root"


def test_slim_root_identity_fields(root_knowledge):
    result = _slim_knowledge_for_prompt(root_knowledge)
    assert result["id"] == root_knowledge.id
    assert result["name"] == root_knowledge.name


def test_slim_root_children_recursively_slimmed(root_knowledge):
    result = _slim_knowledge_for_prompt(root_knowledge)
    assert isinstance(result["children"], list)
    for child in result["children"]:
        assert "type" in child
        assert "id" in child


def test_slim_root_no_verbose_fields(root_knowledge):
    result = _slim_knowledge_for_prompt(root_knowledge)
    for field in ("sources", "description", "bloom_level"):
        assert field not in result


# ---------------------------------------------------------------------------
# ConceptualKnowledge
# ---------------------------------------------------------------------------


def test_slim_conceptual_type(conceptual_knowledge):
    result = _slim_knowledge_for_prompt(conceptual_knowledge)
    assert result["type"] == "conceptual"


def test_slim_conceptual_identity_fields(conceptual_knowledge):
    result = _slim_knowledge_for_prompt(conceptual_knowledge)
    assert result["id"] == conceptual_knowledge.id
    assert result["name"] == conceptual_knowledge.name
    assert result["label"] == conceptual_knowledge.label


def test_slim_conceptual_short_definition_unchanged(conceptual_knowledge_factory):
    node = conceptual_knowledge_factory.build(definition="Short", children=[])
    result = _slim_knowledge_for_prompt(node)
    assert result["definition"] == "Short"
    assert not result["definition"].endswith("…")


def test_slim_conceptual_long_definition_truncated(conceptual_knowledge_factory):
    long_def = "a" * 300
    node = conceptual_knowledge_factory.build(definition=long_def, children=[])
    result = _slim_knowledge_for_prompt(node)
    assert result["definition"] == "a" * 200 + "…"
    assert len(result["definition"]) == 201


def test_slim_conceptual_definition_at_boundary_not_truncated(
    conceptual_knowledge_factory,
):
    exact_def = "b" * 200
    node = conceptual_knowledge_factory.build(definition=exact_def, children=[])
    result = _slim_knowledge_for_prompt(node)
    assert result["definition"] == exact_def
    assert not result["definition"].endswith("…")


def test_slim_conceptual_connections_serialized(conceptual_knowledge_factory):
    connection = models.ConceptualKnowledgeConnection(
        relation=models.KnowledgeConceptualLinkType.DEPENDS_ON,
        to="target_node_id",
    )
    node = conceptual_knowledge_factory.build(connections=[connection], children=[])
    result = _slim_knowledge_for_prompt(node)
    assert result["connections"] == [{"relation": "DEPENDS_ON", "to": "target_node_id"}]


def test_slim_conceptual_verbose_fields_excluded(conceptual_knowledge):
    result = _slim_knowledge_for_prompt(conceptual_knowledge)
    for field in (
        "bloom_level",
        "visibility",
        "validation_status",
        "confidence_score",
        "relevance_score",
        "source",
        "learning_objective",
    ):
        assert field not in result


def test_slim_conceptual_children_recursively_slimmed(conceptual_knowledge):
    result = _slim_knowledge_for_prompt(conceptual_knowledge)
    for child in result["children"]:
        assert "type" in child
        assert "id" in child


# ---------------------------------------------------------------------------
# ProceduralKnowledge
# ---------------------------------------------------------------------------


def test_slim_procedural_type(procedural_knowledge):
    result = _slim_knowledge_for_prompt(procedural_knowledge)
    assert result["type"] == "procedural"


def test_slim_procedural_identity_fields(procedural_knowledge):
    result = _slim_knowledge_for_prompt(procedural_knowledge)
    assert result["id"] == procedural_knowledge.id
    assert result["name"] == procedural_knowledge.name
    assert result["label"] == procedural_knowledge.label
    assert result["percent_done"] == procedural_knowledge.percent_done


def test_slim_procedural_none_child(procedural_knowledge_factory):
    node = procedural_knowledge_factory.build(child=None)
    result = _slim_knowledge_for_prompt(node)
    assert result["child"] is None


def test_slim_procedural_child_recursively_slimmed(procedural_knowledge_factory):
    child_node = procedural_knowledge_factory.build(child=None)
    node = procedural_knowledge_factory.build(child=child_node)
    result = _slim_knowledge_for_prompt(node)
    assert result["child"]["type"] == "procedural"
    assert result["child"]["id"] == child_node.id


def test_slim_procedural_verbose_fields_excluded(procedural_knowledge):
    result = _slim_knowledge_for_prompt(procedural_knowledge)
    for field in (
        "bloom_level",
        "common_errors",
        "visibility",
        "source",
        "learning_objective",
        "validation_status",
        "confidence_score",
        "relevance_score",
    ):
        assert field not in result


# ---------------------------------------------------------------------------
# AssessmentKnowledge
# ---------------------------------------------------------------------------


def test_slim_assessment_type(assessment_knowledge):
    result = _slim_knowledge_for_prompt(assessment_knowledge)
    assert result["type"] == "assessment"


def test_slim_assessment_identity_fields(assessment_knowledge):
    result = _slim_knowledge_for_prompt(assessment_knowledge)
    assert result["id"] == assessment_knowledge.id
    assert result["name"] == assessment_knowledge.name
    assert result["label"] == assessment_knowledge.label


def test_slim_assessment_linked_challenges_included(assessment_knowledge_factory):
    node = assessment_knowledge_factory.build(linked_challenges=["ch_1", "ch_2"])
    result = _slim_knowledge_for_prompt(node)
    assert result["linked_challenges"] == ["ch_1", "ch_2"]


def test_slim_assessment_objectives_included(assessment_knowledge_factory):
    node = assessment_knowledge_factory.build(
        objectives=["Load binary", "Solve puzzle"]
    )
    result = _slim_knowledge_for_prompt(node)
    assert result["objectives"] == ["Load binary", "Solve puzzle"]


def test_slim_assessment_verbose_fields_excluded(assessment_knowledge):
    result = _slim_knowledge_for_prompt(assessment_knowledge)
    for field in ("bloom_level", "question_prompts", "evaluation_criteria"):
        assert field not in result
