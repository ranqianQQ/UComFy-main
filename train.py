import copy
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from models import GCN, UComFyGCN


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return float("nan")
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().item())


@torch.no_grad()
def evaluate(model, data, mask, confidence=None, base_edge_weight=None) -> Dict[str, float]:
    model.eval()
    logits = model(data.x, data.edge_index, confidence=confidence, base_edge_weight=base_edge_weight) \
        if isinstance(model, UComFyGCN) else model(data.x, data.edge_index, edge_weight=base_edge_weight)
    loss = F.cross_entropy(logits[mask], data.y[mask]).item() if int(mask.sum()) > 0 else float("nan")
    return {"loss": loss, "acc": accuracy(logits[mask], data.y[mask])}


def build_optimizer(model, lr: float, weight_decay: float, ungsl_lr: float):
    if isinstance(model, UComFyGCN):
        return torch.optim.Adam(
            [
                {"params": list(model.gcn_parameters()), "lr": lr, "weight_decay": weight_decay},
                {"params": list(model.ungsl_parameters()), "lr": ungsl_lr, "weight_decay": 0.0},
            ]
        )
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_model(
    data,
    model,
    train_mask,
    val_mask,
    test_mask,
    epochs: int,
    lr: float,
    weight_decay: float,
    ungsl_lr: float,
    confidence: Optional[torch.Tensor] = None,
    base_edge_weight: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    optimizer = build_optimizer(model, lr, weight_decay, ungsl_lr)
    best_state = None
    best_val = -1.0
    best_test = -1.0
    start = time.time()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, confidence=confidence, base_edge_weight=base_edge_weight) \
            if isinstance(model, UComFyGCN) else model(data.x, data.edge_index, edge_weight=base_edge_weight)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        val_result = evaluate(model, data, val_mask, confidence=confidence, base_edge_weight=base_edge_weight)
        test_result = evaluate(model, data, test_mask, confidence=confidence, base_edge_weight=base_edge_weight)
        if val_result["acc"] >= best_val:
            best_val = val_result["acc"]
            best_test = test_result["acc"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_val_acc": best_val * 100.0,
        "test_acc_at_best_val": best_test * 100.0,
        "train_time": time.time() - start,
    }


def pretrain_gcn(
    data,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
    dropout: float,
    train_mask,
    val_mask,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> torch.Tensor:
    model = GCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=dropout,
    ).to(data.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val = -1.0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        val_acc = evaluate(model, data, val_mask)["acc"]
        if val_acc >= best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(data.x, data.edge_index).detach()
