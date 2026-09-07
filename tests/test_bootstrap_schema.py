"""Tests for bootstrap_schema (#1510).

Covers the draft-induction contract: it wraps the existing
``OntologyGenerator`` pipeline, threads ``min_occurrences`` through as a
frequency gate, and emits a draft TTL that is never auto-applied.
"""

from semantica.ontology import bootstrap_schema


def _sample(dup: bool = True):
    """A small extraction sample with a common 'Person'/'worksFor' core."""
    entities = [
        {"type": "Person", "text": "John"},
        {"type": "Person", "text": "Jane"},
        {"type": "Org", "text": "Acme"},
        {"type": "Product", "text": "Widget"},
    ]
    rels = [
        {"type": "worksFor", "source": "John", "target": "Acme"},
    ]
    if dup:
        rels.append({"type": "worksFor", "source": "Jane", "target": "Acme"})
    return entities, rels


def test_returns_a_draft_never_auto_applied():
    entities, rels = _sample()
    result = bootstrap_schema(entities, rels)
    assert result["draft"] is True


def test_induces_classes_and_properties_with_domain_range():
    entities, rels = _sample()
    result = bootstrap_schema(entities, rels)
    ontology = result["ontology"]
    class_names = {c.get("name") for c in ontology.get("classes", [])}
    assert "Person" in class_names
    props = ontology.get("properties", [])
    assert any(
        p.get("name") == "worksFor"
        and "Person" in p.get("domain", [])
        and "Org" in p.get("range", [])
        for p in props
    )


def test_min_occurrences_gates_sparse_types():
    # 'Product' appears once; with a higher gate it should be dropped from
    # the draft's classes, while the frequent core survives.
    entities, rels = _sample()
    result = bootstrap_schema(entities, rels, min_occurrences=2)
    class_names = {c.get("name") for c in result["ontology"].get("classes", [])}
    assert "Product" not in class_names
    assert "Person" in class_names


def test_min_occurrences_gates_sparse_predicates_too():
    # A predicate seen fewer than min_occurrences is dropped from the draft's
    # properties — not just types. This is the contract clarified against
    # #1510's "class inference only" framing.
    entities = _sample()[0]
    # Two worksFor (frequent) and a single locatedIn (sparse).
    rels = [
        {"type": "worksFor", "source": "John", "target": "Acme"},
        {"type": "worksFor", "source": "Jane", "target": "Acme"},
        {"type": "locatedIn", "source": "John", "target": "Acme"},
    ]
    result = bootstrap_schema(entities, rels, min_occurrences=2)
    prop_names = {p.get("name") for p in result["ontology"].get("properties", [])}
    assert "worksFor" in prop_names
    assert "locatedIn" not in prop_names


def test_emits_ttl_for_human_review():
    entities, rels = _sample()
    result = bootstrap_schema(entities, rels)
    ttl = result["ttl"]
    assert isinstance(ttl, str) and len(ttl) > 0
    assert "@prefix" in ttl


def test_exposes_min_occurrences_used():
    entities, rels = _sample()
    result = bootstrap_schema(entities, rels, min_occurrences=3)
    assert result["min_occurrences"] == 3


def test_empty_input_returns_empty_draft():
    # Nothing extracted yet: still a valid (empty) draft with a parseable
    # TTL, so callers can run it early in the pipeline without guarding.
    result = bootstrap_schema([], [])
    assert result["draft"] is True
    assert result["ontology"]["classes"] == []
    assert result["ontology"]["properties"] == []
    assert "@prefix" in result["ttl"]