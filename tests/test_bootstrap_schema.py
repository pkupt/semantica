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
        {"type": "Org", "text": "Globex"},
        {"type": "Product", "text": "Widget"},
    ]
    rels = [
        {"type": "worksFor", "source": "John", "target": "Acme"},
        {"type": "worksFor", "source": "Jane", "target": "Globex"},
    ]
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


def test_owl_thing_serializes_as_standard_owl_iri():
    # An unresolved relationship endpoint gets owl:Thing; serializing it must
    # mint the standard OWL IRI, not a generated local "Thing" class that would
    # over-constrain the draft and reject later typed relationships.
    entities, rels = _sample()
    # Two worksFor (passes the predicate gate) whose range endpoint is not an
    # entity type, so the range stays unresolved -> owl:Thing.
    rels = [
        {"type": "worksFor", "source": "John", "target": "SomeStranger"},
        {"type": "worksFor", "source": "Jane", "target": "AnotherStranger"},
    ]
    result = bootstrap_schema(entities, rels)
    # The standard OWL term must appear (rdflib writes it as the qname prefix),
    # and the ontology must not mint a local Thing class under its own base.
    assert "owl:Thing" in result["ttl"], (
        "owl:Thing must serialize to the standard OWL term"
    )
    assert "class/Thing" not in result["ttl"]
    from rdflib import Graph, Namespace
    from rdflib.namespace import OWL

    g = Graph().parse(data=result["ttl"], format="turtle")
    base = Namespace(result["ontology"]["uri"].rstrip("/") + "/")
    local_things = list(g.subjects(OWL.Class, None))
    assert base["Thing"] not in local_things, "must not mint a local Thing class"


def test_sparse_endpoint_types_not_folded_back_by_from_ontology():
    # A rare type that rides in on a frequent relationship's endpoint must not
    # be smuggled back into the concept vocabulary via from_ontology: the
    # frequency gate should stay decisive.
    from semantica.semantic_extract.schema import ExtractionSchema

    entities, rels = _sample()
    # 'Product' appears once (fails class gate) but is a worksFor endpoint.
    rels = [
        {"type": "worksFor", "source": "John", "target": "Widget"},
        {"type": "worksFor", "source": "Jane", "target": "Widget"},
    ]
    result = bootstrap_schema(entities, rels, min_occurrences=2)
    ontology = result["ontology"]
    class_names = {c.get("name") for c in ontology.get("classes", [])}
    assert "Product" not in class_names
    schema = ExtractionSchema.from_ontology(ontology)
    assert "Product" not in schema.concepts


def test_ttl_iris_consistent_with_declared_classes():
    # Finding 2: domain/range endpoints must resolve to the same IRIs as the
    # declared classes (no split between hashed and speaking IRIs).
    entities, rels = _sample()
    result = bootstrap_schema(entities, rels)
    ttl = result["ttl"]
    for cls in result["ontology"].get("classes", []):
        name = cls.get("name")
        if name:
            assert name in ttl, f"declared class {name} missing from TTL"


def test_per_call_min_occurrences_gates_classes_too():
    # Finding 3: a per-call threshold must gate classes as well as predicates,
    # not silently apply only to predicates.
    from semantica.ontology import OntologyGenerator

    entities, rels = _sample()
    gen = OntologyGenerator()  # constructor default gate = 2
    ontology = gen.generate_ontology(
        {"entities": entities, "relationships": rels}, min_occurrences=3
    )
    class_names = {c.get("name") for c in ontology.get("classes", [])}
    assert "Product" not in class_names, "per-call gate must drop sparse class"