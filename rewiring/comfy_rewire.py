import time
from typing import Dict, List

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from utils.graph_utils import clone_data_with_edge_index, undirected_edge_index_from_edges
from utils.metrics import graph_metrics

from .community_utils import (
    allocate_pair_budgets,
    detect_louvain_communities,
    edges_between_communities,
    lowest_edges_by_similarity,
    normalized_features,
    top_non_edges_by_similarity,
)


def comfy_rewire(
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
    communities = detect_louvain_communities(graph, seed)
    norm_x = normalized_features(data.x)
    add_budgets = allocate_pair_budgets(communities, budget_edges_add)
    delete_budgets = allocate_pair_budgets(communities, budget_edges_delete)

    added = set()
    deleted = set()
    warnings: List[str] = []

    for i, comm_a in enumerate(communities):
        nodes_a = sorted(comm_a)
        for j in range(i, len(communities)):
            comm_b = communities[j]
            nodes_b = sorted(comm_b)
            same = i == j

            delete_k = max(0, delete_budgets.get((i, j), 0))
            if delete_k:
                existing = edges_between_communities(graph, nodes_a, nodes_b, same)
                for u, v in lowest_edges_by_similarity(norm_x, existing, delete_k):
                    if graph.has_edge(u, v) and len(deleted) < budget_edges_delete:
                        graph.remove_edge(u, v)
                        deleted.add((u, v))

            add_k = max(0, add_budgets.get((i, j), 0))
            if add_k:
                candidates = top_non_edges_by_similarity(
                    graph,
                    norm_x,
                    nodes_a,
                    nodes_b,
                    add_k,
                    same,
                    max_non_edges_per_pair,
                    candidate_topk_multiplier,
                    warnings,
                )
                for u, v in candidates:
                    if not graph.has_edge(u, v) and len(added) < budget_edges_add:
                        graph.add_edge(u, v)
                        added.add((u, v))

    edge_index = undirected_edge_index_from_edges(graph.edges(), data.num_nodes)
    rewired_data = clone_data_with_edge_index(data, edge_index)
    metrics_after = graph_metrics(rewired_data.detach().cpu(), seed)

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
