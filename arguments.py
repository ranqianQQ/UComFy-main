import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="UComFy: ComFy rewiring with UnGSL-style uncertainty-aware edge gating."
    )
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--model", type=str, default="UComFyGCN", choices=["GCN", "UComFyGCN"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", type=int, default=1)

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dimension", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--budget_edges_add", type=int, default=100)
    parser.add_argument("--budget_edges_delete", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--init_threshold", type=float, default=0.1)
    parser.add_argument("--ungsl_lr", type=float, default=None)
    parser.add_argument("--normalize_confidence", action="store_true")

    parser.add_argument("--rewire_method", type=str, default="comfy", choices=["comfy", "feast", "none"])
    parser.add_argument("--out", type=str, default="results/ucomfy_results.csv")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--hetero_data_path", type=str, default=None)

    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--cache_entropy", dest="cache_entropy", action="store_true")
    cache_group.add_argument("--no_cache_entropy", dest="cache_entropy", action="store_false")
    parser.set_defaults(cache_entropy=True)

    parser.add_argument("--num_train", type=int, default=20)
    parser.add_argument("--num_val", type=int, default=500)
    parser.add_argument("--largest_connected_component", action="store_true", default=True)
    parser.add_argument("--no_largest_connected_component", dest="largest_connected_component", action="store_false")

    parser.add_argument("--max_non_edges_per_pair", type=int, default=2_000_000)
    parser.add_argument("--candidate_topk_multiplier", type=int, default=20)
    parser.add_argument("--similarity_chunk_size", type=int, default=1024)

    parser.add_argument("--formal_summary_out", type=str, default="results/formal_summary.csv")
    parser.add_argument("--formal_log_out", type=str, default="results/formal_experiment_log.md")
    parser.add_argument("--dataset_status_out", type=str, default="results/dataset_status.md")
    parser.add_argument("--gate_diagnostics_out", type=str, default="results/gate_diagnostics.csv")
    parser.add_argument(
        "--result_type",
        type=str,
        default="auto",
        choices=["auto", "smoke", "pilot", "formal-lite", "medium", "full-formal", "exploratory"],
    )
    parser.add_argument("--rewire_variant", type=str, default="auto")

    args = parser.parse_args()
    if args.ungsl_lr is None:
        args.ungsl_lr = args.lr
    if args.splits < 1:
        parser.error("--splits must be >= 1")
    if args.num_layers < 1:
        parser.error("--num_layers must be >= 1")
    if args.budget_edges_add < 0 or args.budget_edges_delete < 0:
        parser.error("rewiring budgets must be non-negative")
    return args
