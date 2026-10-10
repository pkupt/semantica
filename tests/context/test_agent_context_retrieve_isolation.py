"""Regression tests for issue #1794.

``AgentContext`` reads and writes shared state with its store:

* ``store()`` kept the caller's ``metadata`` / ``entities`` containers by
  reference, so mutating the dict that was passed in rewrote the record.
* ``get_memory()`` and ``retrieve()`` handed back the stored containers, so
  mutating a result rewrote the record -- a read acting as a write channel.
* ``AgentContext.get_memory()`` read the identifier under ``"id"`` while
  ``AgentMemory.get_memory()`` returns it under ``"memory_id"``, so the record
  came back with ``"id": None``.
* ``retrieve(..., anchor_node=..., max_hops=...)`` discarded every result it
  could not place on the graph, returning ``[]`` even though matching records
  existed.
"""

import pytest

from semantica.context import AgentContext, ContextGraph
from semantica.vector_store import VectorStore


def _context() -> AgentContext:
    """Memory path: no knowledge graph, so ``retrieve()`` uses AgentMemory."""
    return AgentContext(
        vector_store=VectorStore(backend="inmemory", dimension=64),
        retention_days=None,
        decision_tracking=False,
        kg_algorithms=False,
        vector_store_features=False,
    )


def _graph_context() -> AgentContext:
    return AgentContext(
        vector_store=VectorStore(backend="inmemory", dimension=64),
        knowledge_graph=ContextGraph(),
        retention_days=None,
        decision_tracking=False,
        kg_algorithms=False,
        vector_store_features=False,
    )


# --- store() must not adopt the caller's containers ----------------------


def test_caller_mutation_after_store_does_not_change_the_record():
    ctx = _context()
    metadata = {"tags": ["a"]}
    memory_id = ctx.store("hello world", metadata=metadata)

    metadata["tags"].append("AFTER_STORE")

    assert ctx.get_memory(memory_id)["metadata"] == {"tags": ["a"]}


def test_store_does_not_inject_filters_into_the_callers_dict():
    ctx = _context()
    metadata = {"tags": ["a"]}

    ctx.store("hello world", metadata=metadata, conversation_id="conv1", user_id="u1")

    assert metadata == {"tags": ["a"]}


# --- reads must return copies -------------------------------------------


def test_mutating_get_memory_result_does_not_change_the_record():
    ctx = _context()
    memory_id = ctx.store("hello world", metadata={"tags": ["a"]})

    ctx.get_memory(memory_id)["metadata"]["tags"].append("FROM_READ")

    assert ctx.get_memory(memory_id)["metadata"] == {"tags": ["a"]}


def test_mutating_a_nested_container_does_not_change_the_record():
    ctx = _context()
    memory_id = ctx.store("hello world", metadata={"nested": {"k": [1]}})

    ctx.get_memory(memory_id)["metadata"]["nested"]["k"].append(2)

    assert ctx.get_memory(memory_id)["metadata"] == {"nested": {"k": [1]}}


def test_mutating_retrieve_result_does_not_change_the_record():
    ctx = _context()
    memory_id = ctx.store("hello world", metadata={"tags": ["a"]})

    hits = ctx.retrieve("hello")
    assert hits, "the record just stored must be retrievable"
    for hit in hits:
        hit["metadata"].setdefault("tags", []).append("FROM_RETRIEVE")

    assert ctx.get_memory(memory_id)["metadata"] == {"tags": ["a"]}


# --- the record must carry its own identifier ---------------------------


def test_get_memory_returns_the_stored_identifier():
    ctx = _context()
    memory_id = ctx.store("hello world", metadata={})

    assert ctx.get_memory(memory_id)["id"] == memory_id


# --- an unplaceable result is kept, not dropped -------------------------


def test_retrieve_with_anchor_and_max_hops_keeps_matching_records():
    ctx = _graph_context()
    ctx.store("hello world", metadata={"tags": ["a"]})

    # "n1" is not in the graph, so no result can be given a hop distance.
    # They used to be dropped whenever max_hops was set, so this call
    # returned [] on a store that clearly had a matching record.
    results = ctx.retrieve("hello", anchor_node="n1", max_hops=2)

    assert "hello world" in [r["content"] for r in results]
