from typing import Dict, Optional

import torch


@torch.no_grad()
def compute_gate_diagnostics(
    model,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    confidence: torch.Tensor,
    base_edge_weight: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    source = edge_index[0]
    target = edge_index[1]
    thresholds = model.thresholds.view(-1)
    raw = torch.sigmoid(confidence[source] - thresholds[target]) / 0.5
    base = torch.ones_like(raw) if base_edge_weight is None else base_edge_weight.to(raw.device).view(-1)
    beta = torch.full_like(raw, float(model.gate.beta))
    gate = torch.where(raw >= 1.0, raw, beta)
    gate = 1.0 + float(model.gate.gate_residual_alpha) * (gate - 1.0)
    edge_weight = base * gate

    same_label = y[source] == y[target]
    diff_label = ~same_label
    same_mean = edge_weight[same_label].mean() if bool(same_label.any()) else torch.tensor(float("nan"), device=edge_weight.device)
    diff_mean = edge_weight[diff_label].mean() if bool(diff_label.any()) else torch.tensor(float("nan"), device=edge_weight.device)

    return {
        "beta_branch_ratio": float((raw < 1.0).float().mean().item()) if raw.numel() else float("nan"),
        "same_label_weight_mean": float(same_mean.item()),
        "diff_label_weight_mean": float(diff_mean.item()),
        "edge_weight_mean": float(edge_weight.mean().item()) if edge_weight.numel() else float("nan"),
        "edge_weight_min": float(edge_weight.min().item()) if edge_weight.numel() else float("nan"),
        "edge_weight_max": float(edge_weight.max().item()) if edge_weight.numel() else float("nan"),
    }
