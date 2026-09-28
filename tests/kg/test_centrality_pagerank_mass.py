"""PageRank is a probability distribution: the scores must sum to 1.

Regression tests for #1759. Part of the mass used to be dropped whenever a
column of the transition matrix was left empty, which happens when a node
filter removes every outgoing neighbour of a node, or when a node simply has
no outgoing edge inside the current node set. The scores then drift below 1
and are no longer comparable across runs.
"""

import networkx as nx
import pytest

from semantica.kg import CentralityCalculator


@pytest.fixture
def calculator():
    return CentralityCalculator()


def _scores(result):
    return result["centrality"]


def test_pagerank_sums_to_one_on_a_directed_graph(calculator):
    """A node with no outgoing edge must not swallow its share of the mass."""
    graph = nx.DiGraph([("alice", "acme"), ("bob", "acme"), ("alice", "bob")])

    scores = _scores(calculator.calculate_pagerank(graph))

    assert sum(scores.values()) == pytest.approx(1.0)
    assert scores == pytest.approx(nx.pagerank(graph))


def test_pagerank_keeps_mass_when_node_filter_drops_a_neighbour(calculator):
    """The weight must be spread over the retained neighbours only (#1759)."""
    graph = nx.Graph()
    graph.add_node("alice", label="person")
    graph.add_node("bob", label="person")
    graph.add_node("acme", label="organization")
    graph.add_edge("alice", "bob")
    graph.add_edge("alice", "acme")

    scores = _scores(calculator.calculate_pagerank(graph, node_labels=["person"]))

    assert sum(scores.values()) == pytest.approx(1.0)
    # Dropping a node has to agree with never adding it (#1759, Expected).
    without = nx.Graph([("alice", "bob")])
    assert scores == pytest.approx(nx.pagerank(without))


def test_pagerank_keeps_mass_when_relationship_filter_drops_edges(calculator):
    graph = nx.Graph()
    for name, label in (("alice", "person"), ("bob", "person"), ("acme", "org")):
        graph.add_node(name, label=label)
    graph.add_edge("alice", "acme", type="works_for")
    graph.add_edge("bob", "acme", type="works_for")
    graph.add_edge("alice", "bob", type="knows")

    scores = _scores(calculator.calculate_pagerank(graph, relationship_types=["knows"]))

    assert sum(scores.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("factory", [nx.Graph, nx.DiGraph])
def test_pagerank_matches_networkx(calculator, factory):
    graph = factory([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])

    # The default cap is 20 iterations, which is not enough for this graph to
    # settle; 200 converges. The remaining ~5e-7 gap is the calculator's own
    # convergence tolerance, so compare with a margin above it.
    scores = _scores(calculator.calculate_pagerank(graph, max_iterations=200))

    assert scores == pytest.approx(nx.pagerank(graph), abs=1e-5)
