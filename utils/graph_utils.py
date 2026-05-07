from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected


def ensure_2d_masks(data: Data) -> Data:
    for name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, name, None)
        if mask is not None and mask.dim() == 1:
            setattr(data, name, mask.view(-1, 1))
    return data


def clone_data_with_edge_index(data: Data, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> Data:
    new_data = data.clone()
    new_data.edge_index = edge_index
    if edge_weight is not None:
        new_data.edge_weight = edge_weight
    elif hasattr(new_data, "edge_weight"):
        delattr(new_data, "edge_weight")
    return new_data


def undirected_edge_index_from_edges(edges, num_nodes: int) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    return edge_index


def mask_for_split(mask: torch.Tensor, split_idx: int) -> torch.Tensor:
    if mask.dim() == 1:
        return mask
    return mask[:, split_idx]
