import os
import sys
import time
from datetime import datetime
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

FORMAL_FIELDS = [
    "timestamp",
    "dataset",
    "method",
    "rewire_method",
    "model",
    "splits",
    "epochs",
    "pretrain_epochs",
    "seed",
    "budget_edges_add",
    "budget_edges_delete",
    "hidden_dimension",
    "num_layers",
    "dropout",
    "lr",
    "weight_decay",
    "beta",
    "init_threshold",
    "ungsl_lr",
    "normalize_confidence",
    "val_acc_mean",
    "val_acc_std",
    "test_acc_mean",
    "test_acc_std",
    "loss_start",
    "loss_end",
    "num_edges_before",
    "num_edges_after",
    "edges_added",
    "edges_deleted",
    "confidence_min",
    "confidence_max",
    "confidence_mean",
    "edge_weight_min",
    "edge_weight_max",
    "edge_weight_mean",
    "threshold_grad_norm",
    "threshold_delta",
    "csv_path",
    "status",
    "notes",
]

FORMAL_FIELDS_GPU = ["result_type", "rewire_variant"] + FORMAL_FIELDS

GATE_DIAGNOSTIC_FIELDS = [
    "timestamp",
    "dataset",
    "method",
    "rewire_variant",
    "result_type",
    "splits",
    "epochs",
    "pretrain_epochs",
    "budget_edges_add",
    "budget_edges_delete",
    "confidence_mean",
    "confidence_minmax",
    "edge_weight_mean",
    "edge_weight_minmax",
    "threshold_delta",
    "threshold_grad_norm",
    "beta_branch_ratio",
    "same_label_weight_mean",
    "diff_label_weight_mean",
    "homophily_before_after",
    "test_acc",
    "notes",
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
        gate_residual_alpha=args.gate_residual_alpha,
    )


def get_confidence_for_split(data, args, split_idx: int, num_features: int, num_classes: int, device):
    import torch

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
            _print_tensor_stats("confidence", confidence)
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
    _print_tensor_stats("entropy", entropy)
    _print_tensor_stats("confidence", confidence)
    if args.cache_entropy:
        save_cached_confidence(cache_path, entropy, confidence)
    return confidence.to(device)


def _print_tensor_stats(name: str, tensor) -> None:
    import torch

    detached = tensor.detach()
    finite = torch.isfinite(detached)
    has_nan = bool(torch.isnan(detached).any().item()) if detached.numel() else False
    has_inf = bool(torch.isinf(detached).any().item()) if detached.numel() else False
    min_val = float(detached.min().item()) if detached.numel() else float("nan")
    max_val = float(detached.max().item()) if detached.numel() else float("nan")
    mean_val = float(detached.mean().item()) if detached.numel() else float("nan")
    print(
        f"[diagnostic] {name}: min={min_val:.6f}, max={max_val:.6f}, "
        f"mean={mean_val:.6f}, finite={bool(finite.all().item()) if detached.numel() else True}, "
        f"nan={has_nan}, inf={has_inf}"
    )
    return {
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "finite": bool(finite.all().item()) if detached.numel() else True,
        "nan": has_nan,
        "inf": has_inf,
    }


def _tensor_stats(tensor) -> Dict[str, object]:
    import torch

    detached = tensor.detach()
    finite = torch.isfinite(detached)
    return {
        "min": float(detached.min().item()) if detached.numel() else float("nan"),
        "max": float(detached.max().item()) if detached.numel() else float("nan"),
        "mean": float(detached.mean().item()) if detached.numel() else float("nan"),
        "finite": bool(finite.all().item()) if detached.numel() else True,
        "nan": bool(torch.isnan(detached).any().item()) if detached.numel() else False,
        "inf": bool(torch.isinf(detached).any().item()) if detached.numel() else False,
    }


def _print_ucomfy_diagnostics(model, result: Dict[str, object]) -> None:
    import torch

    from models import UComFyGCN

    if not isinstance(model, UComFyGCN):
        return
    print(f"[diagnostic] thresholds is nn.Parameter: {isinstance(model.thresholds, torch.nn.Parameter)}")
    print(f"[diagnostic] thresholds.requires_grad: {result['threshold_requires_grad']}")
    print(f"[diagnostic] optimizer contains thresholds: {result['optimizer_has_thresholds']}")
    print(f"[diagnostic] threshold grad norm: {result['threshold_grad_norm']}")
    print(f"[diagnostic] threshold max abs delta after training: {result['threshold_max_abs_delta']}")
    print(f"[diagnostic] UComFyGCN passed edge_weight to GCNConv: {result['used_edge_weight']}")
    stats = result.get("edge_weight_stats")
    if stats:
        print(
            "[diagnostic] edge_weight: "
            f"min={stats['min']:.6f}, max={stats['max']:.6f}, "
            f"mean={stats['mean']:.6f}, finite={stats['finite']}, "
            f"nan={stats.get('nan', False)}, inf={stats.get('inf', False)}"
        )


def _method_name(args) -> str:
    if args.rewire_method == "none" and args.model == "GCN":
        return "GCN"
    if args.rewire_method == "comfy" and args.model == "GCN":
        return "ComFy+GCN"
    if args.rewire_method == "comfy" and args.model == "UComFyGCN":
        return "ComFy+UComFyGCN"
    if args.rewire_method == "feast" and args.model == "GCN":
        return "FeaSt+GCN"
    if args.rewire_method == "feast" and args.model == "UComFyGCN":
        return "FeaSt+UComFyGCN"
    return f"{args.rewire_method}+{args.model}"


def _infer_result_type(args) -> str:
    if args.result_type != "auto":
        return args.result_type
    if args.splits >= 10 and args.epochs >= 100 and args.pretrain_epochs >= 100:
        return "full-formal"
    if args.splits >= 5 and args.epochs >= 100:
        return "medium"
    if args.splits >= 3 and args.epochs >= 50:
        return "formal-lite"
    if args.splits == 1 and args.epochs >= 50:
        return "pilot"
    if args.splits == 1 and args.epochs <= 20:
        return "smoke"
    return "exploratory"


def _infer_rewire_variant(args) -> str:
    if args.rewire_variant != "auto":
        return args.rewire_variant
    if args.rewire_method == "none":
        return "Original"
    if args.rewire_method == "feast":
        return "FeaSt"
    if args.rewire_method == "comfy":
        if args.budget_edges_add > 0 and args.budget_edges_delete > 0:
            return "ComFyAddDel"
        if args.budget_edges_add > 0:
            return "ComFyAdd"
        if args.budget_edges_delete > 0:
            return "ComFyDel"
        return "ComFyZeroBudget"
    return args.rewire_method


def _mean_optional(values):
    import numpy as np

    filtered = [float(value) for value in values if value is not None]
    return float(np.mean(filtered)) if filtered else None


def _aggregate_stats(stats):
    import numpy as np

    filtered = [item for item in stats if item]
    if not filtered:
        return {}
    return {
        "min": float(np.min([item["min"] for item in filtered])),
        "max": float(np.max([item["max"] for item in filtered])),
        "mean": float(np.mean([item["mean"] for item in filtered])),
    }


def _fmt_optional(value):
    return "" if value is None else f"{float(value):.6f}"


def _append_formal_records(
    args,
    metadata,
    num_splits,
    row,
    losses,
    confidence_stats,
    edge_weight_stats,
    threshold_grads,
    threshold_deltas,
    gate_diagnostics,
    notes,
):
    from utils.logging_utils import append_csv, ensure_parent_dir

    method = _method_name(args)
    result_type = _infer_result_type(args)
    rewire_variant = _infer_rewire_variant(args)
    notes_parts = [notes] if notes else []
    if args.model == "UComFyGCN":
        notes_parts.append(f"gate_residual_alpha={args.gate_residual_alpha}")
    notes = "; ".join(notes_parts)
    confidence_agg = _aggregate_stats(confidence_stats)
    edge_weight_agg = _aggregate_stats(edge_weight_stats)
    formal_row = {
        "result_type": result_type,
        "rewire_variant": rewire_variant,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "method": method,
        "rewire_method": args.rewire_method,
        "model": args.model,
        "splits": num_splits,
        "epochs": args.epochs,
        "pretrain_epochs": args.pretrain_epochs,
        "seed": args.seed,
        "budget_edges_add": args.budget_edges_add,
        "budget_edges_delete": args.budget_edges_delete,
        "hidden_dimension": args.hidden_dimension,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "beta": args.beta,
        "init_threshold": args.init_threshold,
        "ungsl_lr": args.ungsl_lr,
        "normalize_confidence": args.normalize_confidence,
        "val_acc_mean": row["avg_val_acc"],
        "val_acc_std": row["std_val_acc"],
        "test_acc_mean": row["avg_test_acc"],
        "test_acc_std": row["std_test_acc"],
        "loss_start": _fmt_optional(_mean_optional(losses["start"])),
        "loss_end": _fmt_optional(_mean_optional(losses["end"])),
        "num_edges_before": metadata["num_edges_before"],
        "num_edges_after": metadata["num_edges_after"],
        "edges_added": metadata["edges_added"],
        "edges_deleted": metadata["edges_deleted"],
        "confidence_min": _fmt_optional(confidence_agg.get("min")),
        "confidence_max": _fmt_optional(confidence_agg.get("max")),
        "confidence_mean": _fmt_optional(confidence_agg.get("mean")),
        "edge_weight_min": _fmt_optional(edge_weight_agg.get("min")),
        "edge_weight_max": _fmt_optional(edge_weight_agg.get("max")),
        "edge_weight_mean": _fmt_optional(edge_weight_agg.get("mean")),
        "threshold_grad_norm": _fmt_optional(_mean_optional(threshold_grads)),
        "threshold_delta": _fmt_optional(_mean_optional(threshold_deltas)),
        "csv_path": args.out,
        "status": "success",
        "notes": notes,
    }
    summary_fields = FORMAL_FIELDS_GPU if os.path.basename(args.formal_summary_out).endswith("_gpu.csv") else FORMAL_FIELDS
    if summary_fields is FORMAL_FIELDS:
        formal_row = {key: formal_row[key] for key in FORMAL_FIELDS}
    append_csv(args.formal_summary_out, summary_fields, formal_row)

    gate_agg = _aggregate_gate_diagnostics(gate_diagnostics)
    if gate_agg:
        gate_row = {
            "timestamp": formal_row.get("timestamp", datetime.now().isoformat(timespec="seconds")),
            "dataset": args.dataset,
            "method": method,
            "rewire_variant": rewire_variant,
            "result_type": result_type,
            "splits": num_splits,
            "epochs": args.epochs,
            "pretrain_epochs": args.pretrain_epochs,
            "budget_edges_add": args.budget_edges_add,
            "budget_edges_delete": args.budget_edges_delete,
            "confidence_mean": _fmt_optional(confidence_agg.get("mean")),
            "confidence_minmax": (
                f"{_fmt_optional(confidence_agg.get('min'))}/{_fmt_optional(confidence_agg.get('max'))}"
            ),
            "edge_weight_mean": _fmt_optional(edge_weight_agg.get("mean")),
            "edge_weight_minmax": (
                f"{_fmt_optional(edge_weight_agg.get('min'))}/{_fmt_optional(edge_weight_agg.get('max'))}"
            ),
            "threshold_delta": _fmt_optional(_mean_optional(threshold_deltas)),
            "threshold_grad_norm": _fmt_optional(_mean_optional(threshold_grads)),
            "beta_branch_ratio": _fmt_optional(gate_agg.get("beta_branch_ratio")),
            "same_label_weight_mean": _fmt_optional(gate_agg.get("same_label_weight_mean")),
            "diff_label_weight_mean": _fmt_optional(gate_agg.get("diff_label_weight_mean")),
            "homophily_before_after": f"{metadata['homophily_before']}/{metadata['homophily_after']}",
            "test_acc": row["avg_test_acc"],
            "notes": notes,
        }
        append_csv(args.gate_diagnostics_out, GATE_DIAGNOSTIC_FIELDS, gate_row)

    log_path = args.formal_log_out
    ensure_parent_dir(log_path)
    with open(log_path, "a", encoding="utf-8") as handle:
        timestamp = formal_row.get("timestamp", datetime.now().isoformat(timespec="seconds"))
        handle.write(f"\n## {timestamp} {args.dataset} {method} ({result_type}, {rewire_variant})\n\n")
        handle.write("Command:\n\n")
        handle.write("```text\n")
        handle.write(f"{sys.executable} {' '.join(sys.argv)}\n")
        handle.write("```\n\n")
        handle.write(
            f"Result: val {row['avg_val_acc']} +/- {row['std_val_acc']}, "
            f"test {row['avg_test_acc']} +/- {row['std_test_acc']}.\n\n"
        )
        if args.model == "UComFyGCN":
            handle.write(f"Gate residual alpha: {args.gate_residual_alpha}.\n\n")
        handle.write(
            f"Rewiring: before={metadata['num_edges_before']}, after={metadata['num_edges_after']}, "
            f"added={metadata['edges_added']}, deleted={metadata['edges_deleted']}.\n\n"
        )
        if confidence_agg:
            handle.write(
                f"Confidence: min={confidence_agg['min']:.6f}, max={confidence_agg['max']:.6f}, "
                f"mean={confidence_agg['mean']:.6f}.\n\n"
            )
        if edge_weight_agg:
            handle.write(
                f"Edge weight: min={edge_weight_agg['min']:.6f}, max={edge_weight_agg['max']:.6f}, "
                f"mean={edge_weight_agg['mean']:.6f}.\n\n"
            )
        if gate_agg:
            handle.write(
                f"Gate diagnostics: beta_branch_ratio={gate_agg['beta_branch_ratio']:.6f}, "
                f"same_label_weight_mean={gate_agg['same_label_weight_mean']:.6f}, "
                f"diff_label_weight_mean={gate_agg['diff_label_weight_mean']:.6f}.\n\n"
            )
        if notes:
            handle.write(f"Notes: {notes}\n\n")


def _aggregate_gate_diagnostics(items):
    import numpy as np

    filtered = [item for item in items if item]
    if not filtered:
        return {}
    keys = ["beta_branch_ratio", "same_label_weight_mean", "diff_label_weight_mean"]
    return {
        key: float(np.nanmean([item.get(key, float("nan")) for item in filtered]))
        for key in keys
    }


def _append_dataset_status(args, data, metadata, num_features, num_classes, num_splits):
    from utils.logging_utils import ensure_parent_dir

    source = os.path.join(args.hetero_data_path or args.data_root, args.dataset)
    if args.dataset in {"Cora", "Citeseer", "Pubmed"}:
        source = os.path.join(args.data_root, args.dataset)
    split_policy = "LargestConnectedComponents + RandomNodeSplit(train_rest, val=0.2, test=0.2)"
    path = args.dataset_status_out
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"\n## {datetime.now().isoformat(timespec='seconds')} {args.dataset}\n\n")
        handle.write(f"- Source/path: `{source}`\n")
        handle.write(f"- Nodes: {data.num_nodes}\n")
        handle.write(f"- Undirected edges before rewiring: {metadata['num_edges_before']}\n")
        handle.write(f"- Features: {num_features}\n")
        handle.write(f"- Classes: {num_classes}\n")
        handle.write(f"- Splits used: {num_splits}\n")
        handle.write(f"- Split policy: {split_policy}\n")
        handle.write("- ComFy-main consistency: yes, for this formal sanity setting.\n")


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
    print(
        "[diagnostic] rewiring: "
        f"num_edges_before={metadata['num_edges_before']}, "
        f"num_edges_after={metadata['num_edges_after']}, "
        f"edges_added={metadata['edges_added']}, "
        f"edges_deleted={metadata['edges_deleted']}"
    )
    if metadata.get("warnings"):
        print(f"Rewiring warning: {metadata['warnings']}")

    rewired_data = rewired_data.to(device)
    num_splits = min(args.splits, rewired_data.train_mask.shape[1])
    val_scores = []
    test_scores = []
    losses = {"start": [], "end": []}
    confidence_stats = []
    edge_weight_stats = []
    threshold_grads = []
    threshold_deltas = []
    gate_diagnostics = []
    total_train_time = 0.0
    full_start = time.time()

    _append_dataset_status(args, data, metadata, num_features, num_classes, num_splits)

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
            confidence_stats.append(_tensor_stats(confidence))

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
            eval_best_val=args.eval_best_val,
        )
        val_scores.append(float(result["best_val_acc"]))
        test_scores.append(float(result["test_acc_at_best_val"]))
        losses["start"].append(result["first_train_loss"])
        losses["end"].append(result["last_train_loss"])
        if result.get("edge_weight_stats"):
            edge_weight_stats.append(result["edge_weight_stats"])
        if result.get("gate_diagnostics"):
            gate_diagnostics.append(result["gate_diagnostics"])
        threshold_grads.append(result.get("threshold_grad_norm"))
        threshold_deltas.append(result.get("threshold_max_abs_delta"))
        total_train_time += float(result["train_time"])
        print(
            f"[diagnostic] train loss: first={result['first_train_loss']:.6f}, "
            f"last={result['last_train_loss']:.6f}"
        )
        _print_ucomfy_diagnostics(model, result)
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
    _append_formal_records(
        args,
        metadata,
        num_splits,
        row,
        losses,
        confidence_stats,
        edge_weight_stats,
        threshold_grads,
        threshold_deltas,
        gate_diagnostics,
        metadata.get("warnings", ""),
    )
    print(f"Saved results to {args.out}")
    print(f"Mean validation accuracy: {np.mean(val_scores):.2f} +/- {np.std(val_scores):.2f}")
    print(f"Mean test accuracy: {np.mean(test_scores):.2f} +/- {np.std(test_scores):.2f}")
    print(f"Final-model train loop time: {total_train_time:.2f}s")


if __name__ == "__main__":
    main()
