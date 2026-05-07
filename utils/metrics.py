from typing import Dict

import networkx as nx
import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score
from torch_geometric.data import Data
from torch_geometric.utils import homophily, to_networkx


def _as_numpy_labels(labels: torch.Tensor) -> np.ndarray:
    return labels.detach().cpu().numpy()


def edge_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    if edge_index.numel() == 0:
        return float("nan")
    return float(homophily(edge_index.detach().cpu(), y.detach().cpu(), method="edge"))


def adjusted_homophily_from_graph(graph: nx.Graph, labels: np.ndarray) -> float:
    if graph.number_of_edges() == 0:
        return float("nan")
    unique = np.unique(labels)
    label_map = {label: idx for idx, label in enumerate(unique)}
    mapped = np.array([label_map[label] for label in labels])
    degree_sums = np.zeros(len(unique), dtype=float)
    for node in graph.nodes:
        degree_sums[mapped[node]] += graph.degree(node)
    denom = float((2 * graph.number_of_edges()) ** 2)
    adjust = float(np.sum(degree_sums ** 2 / denom)) if denom > 0 else 0.0
    same = sum(1 for u, v in graph.edges if mapped[u] == mapped[v])
    h_edge = same / graph.number_of_edges()
    if abs(1.0 - adjust) < 1e-12:
        return float("nan")
    return float((h_edge - adjust) / (1.0 - adjust))


def louvain_nmi(graph: nx.Graph, labels: np.ndarray, seed: int) -> float:
    if graph.number_of_nodes() == 0:
        return float("nan")
    try:
        communities = list(nx.community.louvain_communities(graph, seed=seed))
    except Exception:
        communities = list(nx.community.greedy_modularity_communities(graph))
    cluster = np.zeros(graph.number_of_nodes(), dtype=int)
    for idx, nodes in enumerate(communities):
        for node in nodes:
            cluster[node] = idx
    return float(normalized_mutual_info_score(cluster, labels))


def graph_metrics(data: Data, seed: int) -> Dict[str, float]:
    cpu_data = data.detach().cpu() if hasattr(data, "detach") else data.cpu()
    graph = to_networkx(cpu_data, to_undirected=True)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    labels = _as_numpy_labels(cpu_data.y)
    return {
        "homophily": edge_homophily(cpu_data.edge_index, cpu_data.y),
        "adjusted_homophily": adjusted_homophily_from_graph(graph, labels),
        "nmi": louvain_nmi(graph, labels, seed),
    }
