import os
from typing import Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, Planetoid
from torch_geometric.transforms import LargestConnectedComponents, NormalizeFeatures, RandomNodeSplit

from utils.graph_utils import ensure_2d_masks


PLANETOID = {"Cora", "Citeseer", "Pubmed"}
COAUTHOR = {"CS", "Physics"}
AMAZON = {"Computers", "Photo"}
HETERO_NPZ = {
    "cornell.npz",
    "texas.npz",
    "wisconsin.npz",
    "chameleon_filtered.npz",
    "squirrel_filtered.npz",
    "actor.npz",
}


def _maybe_lcc(data: Data, enabled: bool) -> Data:
    if enabled:
        data = LargestConnectedComponents()(data)
    return data


def _random_split(data: Data, splits: int) -> Data:
    transform = RandomNodeSplit(split="train_rest", num_splits=splits, num_val=0.2, num_test=0.2)
    return transform(data)


def _normalize_mask(mask, num_nodes: int) -> torch.Tensor:
    mask = torch.as_tensor(mask, dtype=torch.bool)
    if mask.dim() == 1:
        return mask.view(num_nodes, 1)
    if mask.shape[0] == num_nodes:
        return mask.contiguous()
    if mask.shape[1] == num_nodes:
        return mask.t().contiguous()
    raise ValueError(f"Mask shape {tuple(mask.shape)} is incompatible with {num_nodes} nodes.")


def _load_npz_dataset(dataset_name: str, hetero_data_path: str) -> Tuple[Data, int, int]:
    if not hetero_data_path:
        raise FileNotFoundError(
            f"{dataset_name} is an npz heterophily dataset. Pass --hetero_data_path pointing to the directory "
            "containing cornell.npz, texas.npz, wisconsin.npz, chameleon_filtered.npz, squirrel_filtered.npz, or actor.npz."
        )
    filepath = os.path.join(hetero_data_path, dataset_name)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find {filepath}. Check --hetero_data_path and --dataset.")

    payload = np.load(filepath, allow_pickle=True)
    required = {"node_features", "node_labels", "edges"}
    missing = required - set(payload.files)
    if missing:
        raise ValueError(f"{filepath} is missing required arrays: {sorted(missing)}")

    x = torch.as_tensor(payload["node_features"], dtype=torch.float)
    y = torch.as_tensor(payload["node_labels"], dtype=torch.long).view(-1)
    edges = torch.as_tensor(payload["edges"], dtype=torch.long)
    if edges.dim() != 2:
        raise ValueError(f"Expected edges to be a 2D array, got shape {tuple(edges.shape)}")
    edge_index = edges.t().contiguous() if edges.shape[1] == 2 else edges.contiguous()

    data = Data(x=x, edge_index=edge_index, y=y)
    if {"train_masks", "val_masks", "test_masks"}.issubset(payload.files):
        data.train_mask = _normalize_mask(payload["train_masks"], x.shape[0])
        data.val_mask = _normalize_mask(payload["val_masks"], x.shape[0])
        data.test_mask = _normalize_mask(payload["test_masks"], x.shape[0])
    else:
        data = _random_split(data, splits=1)

    num_features = int(data.num_features)
    num_classes = int(torch.unique(data.y).numel())
    data.num_classes = num_classes
    return data, num_features, num_classes


def load_dataset(args) -> Tuple[Data, int, int]:
    name = args.dataset
    root = os.path.join(args.data_root, name.replace(".npz", ""))

    if name in PLANETOID:
        dataset = Planetoid(root=root, name=name, transform=NormalizeFeatures())
        data = dataset[0]
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = _maybe_lcc(data, args.largest_connected_component)
        data = _random_split(data, args.splits)
    elif name in COAUTHOR:
        dataset = Coauthor(root=root, name=name, transform=NormalizeFeatures())
        data = _maybe_lcc(dataset[0], args.largest_connected_component)
        data = _random_split(data, args.splits)
        num_features = dataset.num_features
        num_classes = dataset.num_classes
    elif name in AMAZON:
        dataset = Amazon(root=root, name=name, transform=NormalizeFeatures())
        data = _maybe_lcc(dataset[0], args.largest_connected_component)
        data = _random_split(data, args.splits)
        num_features = dataset.num_features
        num_classes = dataset.num_classes
    elif name in HETERO_NPZ or name.endswith(".npz"):
        data, num_features, num_classes = _load_npz_dataset(name, args.hetero_data_path)
        data = _maybe_lcc(data, args.largest_connected_component)
        if data.train_mask.shape[1] < args.splits:
            data = _random_split(data, args.splits)
    else:
        supported = sorted(PLANETOID | COAUTHOR | AMAZON | HETERO_NPZ)
        raise ValueError(f"Unsupported dataset {name}. Supported datasets: {supported}")

    ensure_2d_masks(data)
    if data.train_mask.shape[1] > args.splits:
        data.train_mask = data.train_mask[:, : args.splits]
        data.val_mask = data.val_mask[:, : args.splits]
        data.test_mask = data.test_mask[:, : args.splits]
    data.num_classes = int(num_classes)
    return data, int(num_features), int(num_classes)
