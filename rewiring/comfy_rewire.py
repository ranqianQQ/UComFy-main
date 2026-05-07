import time
from typing import Dict, List

import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from utils.graph_utils import clone_data_with_edge_index, undirected_edge_index_from_edges
from utils.metrics import graph_metrics

from .community_utils import (
    canonical_edge,
    detect_louvain_communities,
    edges_between_communities,
    normalized_features,
    pair_similarity,
)


def _floor_pair_budgets(communities, total_budget: int) -> Dict[tuple, int]:
    pairs = [(i, j) for i in range(len(communities)) for j in range(i, len(communities))]
    if total_budget <= 0 or not pairs:
        return {pair: 0 for pair in pairs}
    scores = np.array([len(communities[i]) * len(communities[j]) for i, j in pairs], dtype=float)
    denom = float(scores.sum())
    if denom <= 0:
        return {pair: 0 for pair in pairs}
    return {pair: int(total_budget * score / denom) for pair, score in zip(pairs, scores)}


def _iter_pair_non_edges(graph: nx.Graph, nodes_a, nodes_b, same: bool):
    if same:
        for left_idx, u in enumerate(nodes_a):
            for v in nodes_a[left_idx + 1:]:
                if not graph.has_edge(u, v):
                    yield canonical_edge(int(u), int(v))
    else:
        for u in nodes_a:
            for v in nodes_b:
                if int(u) == int(v):
                    continue
                if not graph.has_edge(u, v):
                    yield canonical_edge(int(u), int(v))


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
    add_budgets = _floor_pair_budgets(communities, budget_edges_add)
    delete_budgets = _floor_pair_budgets(communities, budget_edges_delete)

    added = set()
    deleted = set()
    warnings: List[str] = []

    for i, comm_a in enumerate(communities):
        nodes_a = sorted(comm_a)
        for j in range(i, len(communities)):
            comm_b = communities[j]
            nodes_b = sorted(comm_b)
            same = i == j

            existing = edges_between_communities(graph, nodes_a, nodes_b, same)
            if not existing:
                continue

            pair_sims = [pair_similarity(norm_x, u, v) for u, v in existing]
            mean_sim = float(np.mean(pair_sims)) if pair_sims else 0.0
            num_edges = len(existing)

            add_k = max(0, add_budgets.get((i, j), 0))
            if add_k:
                add_rank = []
                for u, v in _iter_pair_non_edges(graph, nodes_a, nodes_b, same):
                    sim_uv = pair_similarity(norm_x, u, v)
                    if sim_uv > mean_sim:
                        score = (mean_sim * num_edges + sim_uv) / (num_edges + 1)
                        add_rank.append((score, u, v))
                add_rank.sort(reverse=True)
                for _, u, v in add_rank[:add_k]:
                    if not graph.has_edge(u, v) and len(added) < budget_edges_add:
                        graph.add_edge(u, v)
                        added.add((u, v))

            delete_k = max(0, delete_budgets.get((i, j), 0))
            if delete_k and num_edges > 1:
                remove_rank = []
                for u, v in existing:
                    sim_uv = pair_similarity(norm_x, u, v)
                    if sim_uv < mean_sim:
                        score = (mean_sim * num_edges - sim_uv) / (num_edges - 1)
                        remove_rank.append((score, u, v))
                remove_rank.sort(reverse=True)
                for _, u, v in remove_rank[:delete_k]:
                    if graph.has_edge(u, v) and len(deleted) < budget_edges_delete:
                        graph.remove_edge(u, v)
                        deleted.add(canonical_edge(u, v))

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
