import heapq
import math
from itertools import combinations, product
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch

Edge = Tuple[int, int]


def detect_louvain_communities(graph: nx.Graph, seed: int) -> List[set]:
    try:
        import nx_cugraph as nxcg  # type: ignore

        backend_graph = nxcg.from_networkx(graph)
        return [set(nodes) for nodes in nx.community.louvain_communities(backend_graph, seed=seed)]
    except Exception:
        return [set(nodes) for nodes in nx.community.louvain_communities(graph, seed=seed)]


def node_to_community(communities: Sequence[set]) -> Dict[int, int]:
    return {node: idx for idx, nodes in enumerate(communities) for node in nodes}


def allocate_pair_budgets(communities: Sequence[set], total_budget: int) -> Dict[Tuple[int, int], int]:
    pairs = [(i, j) for i in range(len(communities)) for j in range(i, len(communities))]
    if total_budget <= 0 or not pairs:
        return {pair: 0 for pair in pairs}
    weights = np.array([len(communities[i]) * len(communities[j]) for i, j in pairs], dtype=float)
    if weights.sum() <= 0:
        return {pair: 0 for pair in pairs}
    raw = total_budget * weights / weights.sum()
    floors = np.floor(raw).astype(int)
    leftover = int(total_budget - floors.sum())
    order = np.argsort(-(raw - floors))
    for idx in order[:leftover]:
        floors[idx] += 1
    return {pair: int(value) for pair, value in zip(pairs, floors)}


def normalized_features(x: torch.Tensor) -> np.ndarray:
    dense = x.detach().cpu().float()
    norms = dense.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (dense / norms).numpy()


def pair_similarity(norm_x: np.ndarray, u: int, v: int) -> float:
    return float(np.dot(norm_x[u], norm_x[v]))


def canonical_edge(u: int, v: int) -> Edge:
    return (u, v) if u <= v else (v, u)


def edges_between_communities(graph: nx.Graph, comm_a: Iterable[int], comm_b: Iterable[int], same: bool) -> List[Edge]:
    set_a = set(comm_a)
    set_b = set(comm_b)
    out = []
    for u, v in graph.edges():
        if same:
            if u in set_a and v in set_a:
                out.append(canonical_edge(u, v))
        elif (u in set_a and v in set_b) or (u in set_b and v in set_a):
            out.append(canonical_edge(u, v))
    return list(dict.fromkeys(out))


def lowest_edges_by_similarity(norm_x: np.ndarray, edges: Sequence[Edge], k: int) -> List[Edge]:
    if k <= 0 or not edges:
        return []
    ranked = sorted((pair_similarity(norm_x, u, v), u, v) for u, v in edges)
    return [(u, v) for _, u, v in ranked[:k]]


def _candidate_count(nodes_a: Sequence[int], nodes_b: Sequence[int], same: bool) -> int:
    if same:
        n = len(nodes_a)
        return n * (n - 1) // 2
    return len(nodes_a) * len(nodes_b)


def _push_top(heap, k: int, score: float, u: int, v: int) -> None:
    item = (score, int(u), int(v))
    if len(heap) < k:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)


def _iter_non_edges(graph: nx.Graph, nodes_a: Sequence[int], nodes_b: Sequence[int], same: bool):
    iterator = combinations(nodes_a, 2) if same else product(nodes_a, nodes_b)
    for u, v in iterator:
        if u == v:
            continue
        if not graph.has_edge(u, v):
            yield canonical_edge(int(u), int(v))


def top_non_edges_by_similarity(
    graph: nx.Graph,
    norm_x: np.ndarray,
    nodes_a: Sequence[int],
    nodes_b: Sequence[int],
    k: int,
    same: bool,
    max_non_edges_per_pair: int,
    candidate_topk_multiplier: int,
    warnings: List[str],
) -> List[Edge]:
    if k <= 0:
        return []

    pair_count = _candidate_count(nodes_a, nodes_b, same)
    heap = []
    if pair_count <= max_non_edges_per_pair:
        for u, v in _iter_non_edges(graph, nodes_a, nodes_b, same):
            _push_top(heap, k, pair_similarity(norm_x, u, v), u, v)
    else:
        warnings.append(
            f"Skipped full non-edge enumeration for a community pair with {pair_count} possible pairs; "
            "used row-wise top-k candidates instead."
        )
        nodes_b_arr = np.array(nodes_b, dtype=np.int64)
        per_row = max(8, math.ceil(k * max(1, candidate_topk_multiplier) / max(1, len(nodes_a))))
        seen = set()
        for u in nodes_a:
            sims = norm_x[int(u)] @ norm_x[nodes_b_arr].T
            local_k = min(len(nodes_b_arr), per_row)
            if local_k == 0:
                continue
            top_idx = np.argpartition(-sims, kth=local_k - 1)[:local_k]
            for idx in top_idx:
                v = int(nodes_b_arr[idx])
                if same and v <= int(u):
                    continue
                if int(u) == v or graph.has_edge(int(u), v):
                    continue
                edge = canonical_edge(int(u), v)
                if edge in seen:
                    continue
                seen.add(edge)
                _push_top(heap, k, float(sims[idx]), edge[0], edge[1])

    ranked = sorted(heap, reverse=True)
    return [(u, v) for _, u, v in ranked]
