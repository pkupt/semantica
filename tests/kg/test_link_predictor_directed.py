"""Directed link prediction (#1941 review).

The undirected candidate set scores each node pair once and drops the pair when
an edge exists in either direction, so on a directed graph an existing A->B
hides a missing B->A. ``predict_links(directed=True)`` scores both orders and
only skips an edge that already exists in the same direction. The default
(``directed=None``) keeps the undirected behaviour so existing callers do not
change.
"""

import networkx as nx

from semantica.kg import LinkPredictor


def _chain() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C")])
    return graph


def test_directed_true_scores_the_missing_reverse_edge():
    predictor = LinkPredictor(method="preferential_attachment")

    undirected = {(a, b) for a, b, _ in predictor.predict_links(_chain(), top_k=10)}
    directed = {
        (a, b)
        for a, b, _ in predictor.predict_links(_chain(), top_k=10, directed=True)
    }

    # A->B exists, so the undirected candidate set never considers B->A.
    assert ("B", "A") not in undirected
    assert ("B", "A") in directed
    # Both orders of the still-missing pair are scored when directed.
    assert ("C", "A") in directed and ("A", "C") in directed


def test_directed_none_matches_the_default_behaviour():
    predictor = LinkPredictor(method="preferential_attachment")
    graph = _chain()

    default = {(a, b) for a, b, _ in predictor.predict_links(graph, top_k=10)}
    explicit_none = {
        (a, b)
        for a, b, _ in predictor.predict_links(graph, top_k=10, directed=None)
    }

    assert explicit_none == default
