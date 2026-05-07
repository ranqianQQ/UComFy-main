import time
from typing import Dict, List

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from utils.graph_utils import clone_data_with_edge_index, undirected_edge_index_from_edges
from utils.metrics import graph_metrics

from .community_utils import (
    detect_louvain_communities,
    lowest_edges_by_similarity,
    normalized_features,
    top_non_edges_by_similarity,
)


def feast_rewire(
    data: Data,
    budget_edges_add: int,
    budget_edges_delete: int,
    seed: int,
    max_non_edges_per_pair: int = 2_000_000,
    candidate_topk_multiplier: int = 20,
) -> (Data, Dict[str, object]):
    start = time.time()
    graph = to_networkx(data.detach().cpu(), to_undirected=True)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph.add_nodes_from(range(data.num_nodes))

    metrics_before = graph_metrics(data.detach().cpu(), seed)
    original_edges = graph.number_of_edges()
    norm_x = normalized_features(data.x)
    warnings: List[str] = []

    deleted = set()
    existing = [(min(u, v), max(u, v)) for u, v in graph.edges()]
    for u, v in lowest_edges_by_similarity(norm_x, existing, budget_edges_delete):
        if graph.has_edge(u, v):
            graph.remove_edge(u, v)
            deleted.add((u, v))

    added = set()
    nodes = list(range(data.num_nodes))
    candidates = top_non_edges_by_similarity(
        graph,
        norm_x,
        nodes,
        nodes,
        budget_edges_add,
        same=True,
        max_non_edges_per_pair=max_non_edges_per_pair,
        candidate_topk_multiplier=candidate_topk_multiplier,
        warnings=warnings,
    )
    for u, v in candidates:
        if not graph.has_edge(u, v):
            graph.add_edge(u, v)
            added.add((u, v))

    edge_index = undirected_edge_index_from_edges(graph.edges(), data.num_nodes)
    rewired_data = clone_data_with_edge_index(data, edge_index)
    metrics_after = graph_metrics(rewired_data.detach().cpu(), seed)
    communities = detect_louvain_communities(graph, seed)

    metadata: Dict[str, object] = {
        "num_edges_before": original_edges,
        "num_edges_after": graph.number_of_edges(),
        "edges_added": len(added),
        "edges_deleted": len(deleted),
        "num_communities": len(communities),
        "rewire_time": time.time() - start,
        "homophily_before": metrics_before["homophily"],
        "homophily_after": metrics_after["homophily"],
        "adjusted_homophily_before": metrics_before["adjusted_homophily"],
        "adjusted_homophily_after": metrics_after["adjusted_homophily"],
        "nmi_before": metrics_before["nmi"],
        "nmi_after": metrics_after["nmi"],
        "warnings": "; ".join(dict.fromkeys(warnings)),
    }
    return rewired_data, metadata
