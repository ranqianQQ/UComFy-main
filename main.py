import os
import time
from typing import Dict

from arguments import parse_args


CSV_FIELDS = [
    "dataset",
    "model",
    "rewire_method",
    "seed",
    "split",
    "avg_val_acc",
    "std_val_acc",
    "avg_test_acc",
    "std_test_acc",
    "budget_edges_add",
    "budget_edges_delete",
    "beta",
    "init_threshold",
    "lr",
    "ungsl_lr",
    "hidden_dimension",
    "dropout",
    "pretrain_epochs",
    "epochs",
    "num_edges_before",
    "num_edges_after",
    "edges_added",
    "edges_deleted",
    "rewire_time",
    "train_time",
    "homophily_before",
    "homophily_after",
    "adjusted_homophily_before",
    "adjusted_homophily_after",
    "nmi_before",
    "nmi_after",
]


def resolve_device(device_name: str):
    import torch

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def no_rewire_metadata(data, seed: int) -> Dict[str, object]:
    import networkx as nx
    from torch_geometric.utils import to_networkx

    from utils.metrics import graph_metrics

    graph = to_networkx(data.detach().cpu(), to_undirected=True)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    metrics = graph_metrics(data.detach().cpu(), seed)
    return {
        "num_edges_before": graph.number_of_edges(),
        "num_edges_after": graph.number_of_edges(),
        "edges_added": 0,
        "edges_deleted": 0,
        "num_communities": 0,
        "rewire_time": 0.0,
        "homophily_before": metrics["homophily"],
        "homophily_after": metrics["homophily"],
        "adjusted_homophily_before": metrics["adjusted_homophily"],
        "adjusted_homophily_after": metrics["adjusted_homophily"],
        "nmi_before": metrics["nmi"],
        "nmi_after": metrics["nmi"],
        "warnings": "",
    }


def run_rewiring(data, args):
    from rewiring import comfy_rewire, feast_rewire

    if args.rewire_method == "none":
        return data, no_rewire_metadata(data, args.seed)
    if args.rewire_method == "comfy":
        return comfy_rewire(
            data,
            args.budget_edges_add,
            args.budget_edges_delete,
            args.seed,
            max_non_edges_per_pair=args.max_non_edges_per_pair,
            candidate_topk_multiplier=args.candidate_topk_multiplier,
        )
    if args.rewire_method == "feast":
        return feast_rewire(
            data,
            args.budget_edges_add,
            args.budget_edges_delete,
            args.seed,
            max_non_edges_per_pair=args.max_non_edges_per_pair,
            candidate_topk_multiplier=args.candidate_topk_multiplier,
        )
    raise ValueError(f"Unknown rewire method {args.rewire_method}")


def build_model(args, num_features: int, num_classes: int, num_nodes: int):
    from models import GCN, UComFyGCN

    if args.model == "GCN":
        return GCN(
            in_channels=num_features,
            hidden_channels=args.hidden_dimension,
            out_channels=num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    return UComFyGCN(
        in_channels=num_features,
        hidden_channels=args.hidden_dimension,
        out_channels=num_classes,
        num_nodes=num_nodes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        beta=args.beta,
        init_threshold=args.init_threshold,
    )


def get_confidence_for_split(data, args, split_idx: int, num_features: int, num_classes: int, device):
    from modules.uncertainty import (
        confidence_cache_path,
        confidence_from_logits,
        load_cached_confidence,
        save_cached_confidence,
    )
    from train import pretrain_gcn
    from utils.graph_utils import mask_for_split

    cache_dir = os.path.join("results", "cache")
    cache_path = confidence_cache_path(
        cache_dir,
        args.dataset,
        args.seed,
        split_idx,
        args.rewire_method,
        args.budget_edges_add,
        args.budget_edges_delete,
        args.normalize_confidence,
    )
    if args.cache_entropy:
        entropy, confidence = load_cached_confidence(cache_path, device)
        if entropy is not None and confidence is not None:
            return confidence.to(device)

    train_mask = mask_for_split(data.train_mask, split_idx)
    val_mask = mask_for_split(data.val_mask, split_idx)
    logits = pretrain_gcn(
        data,
        in_channels=num_features,
        hidden_channels=args.hidden_dimension,
        out_channels=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        train_mask=train_mask,
        val_mask=val_mask,
        epochs=args.pretrain_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    entropy, confidence = confidence_from_logits(logits, normalize=args.normalize_confidence)
    if args.cache_entropy:
        save_cached_confidence(cache_path, entropy, confidence)
    return confidence.to(device)


def main():
    args = parse_args()
    import numpy as np
    import torch

    from dataloader import load_dataset
    from train import train_model
    from utils.graph_utils import mask_for_split
    from utils.logging_utils import append_csv
    from utils.seed import set_seed

    set_seed(args.seed)
    device = resolve_device(args.device)

    print(f"Loading dataset {args.dataset}...")
    data, num_features, num_classes = load_dataset(args)
    print(data)

    print(f"Running rewiring method: {args.rewire_method}")
    rewired_data, metadata = run_rewiring(data, args)
    if metadata.get("warnings"):
        print(f"Rewiring warning: {metadata['warnings']}")

    rewired_data = rewired_data.to(device)
    num_splits = min(args.splits, rewired_data.train_mask.shape[1])
    val_scores = []
    test_scores = []
    total_train_time = 0.0
    full_start = time.time()

    for split_idx in range(num_splits):
        print(f"Split {split_idx + 1}/{num_splits}")
        set_seed(args.seed + split_idx)
        train_mask = mask_for_split(rewired_data.train_mask, split_idx)
        val_mask = mask_for_split(rewired_data.val_mask, split_idx)
        test_mask = mask_for_split(rewired_data.test_mask, split_idx)

        confidence = None
        if args.model == "UComFyGCN":
            confidence = get_confidence_for_split(
                rewired_data,
                args,
                split_idx,
                num_features,
                num_classes,
                device,
            )

        model = build_model(args, num_features, num_classes, rewired_data.num_nodes).to(device)
        result = train_model(
            rewired_data,
            model,
            train_mask,
            val_mask,
            test_mask,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            ungsl_lr=args.ungsl_lr,
            confidence=confidence,
            base_edge_weight=getattr(rewired_data, "edge_weight", None),
        )
        val_scores.append(float(result["best_val_acc"]))
        test_scores.append(float(result["test_acc_at_best_val"]))
        total_train_time += float(result["train_time"])
        print(
            f"Split {split_idx}: val={result['best_val_acc']:.2f}, "
            f"test@best_val={result['test_acc_at_best_val']:.2f}"
        )

    total_wall_train_time = time.time() - full_start
    row = {
        "dataset": args.dataset,
        "model": args.model,
        "rewire_method": args.rewire_method,
        "seed": args.seed,
        "split": "0" if num_splits == 1 else f"0-{num_splits - 1}",
        "avg_val_acc": f"{np.mean(val_scores):.4f}",
        "std_val_acc": f"{np.std(val_scores):.4f}",
        "avg_test_acc": f"{np.mean(test_scores):.4f}",
        "std_test_acc": f"{np.std(test_scores):.4f}",
        "budget_edges_add": args.budget_edges_add,
        "budget_edges_delete": args.budget_edges_delete,
        "beta": args.beta,
        "init_threshold": args.init_threshold,
        "lr": args.lr,
        "ungsl_lr": args.ungsl_lr,
        "hidden_dimension": args.hidden_dimension,
        "dropout": args.dropout,
        "pretrain_epochs": args.pretrain_epochs,
        "epochs": args.epochs,
        "num_edges_before": metadata["num_edges_before"],
        "num_edges_after": metadata["num_edges_after"],
        "edges_added": metadata["edges_added"],
        "edges_deleted": metadata["edges_deleted"],
        "rewire_time": f"{float(metadata['rewire_time']):.4f}",
        "train_time": f"{total_wall_train_time:.4f}",
        "homophily_before": metadata["homophily_before"],
        "homophily_after": metadata["homophily_after"],
        "adjusted_homophily_before": metadata["adjusted_homophily_before"],
        "adjusted_homophily_after": metadata["adjusted_homophily_after"],
        "nmi_before": metadata["nmi_before"],
        "nmi_after": metadata["nmi_after"],
    }
    append_csv(args.out, CSV_FIELDS, row)
    print(f"Saved results to {args.out}")
    print(f"Mean validation accuracy: {np.mean(val_scores):.2f} +/- {np.std(val_scores):.2f}")
    print(f"Mean test accuracy: {np.mean(test_scores):.2f} +/- {np.std(test_scores):.2f}")
    print(f"Final-model train loop time: {total_train_time:.2f}s")


if __name__ == "__main__":
    main()
