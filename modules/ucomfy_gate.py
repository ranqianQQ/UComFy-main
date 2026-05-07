import torch
import torch.nn as nn


class UncertaintyEdgeGate(nn.Module):
    """UnGSL-style directional gate for PyG edge_index tensors."""

    def __init__(self, num_nodes: int, beta: float = 0.2, init_threshold: float = 0.5):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.beta = float(beta)
        self.init_threshold = float(init_threshold)
        self.thresholds = nn.Parameter(torch.empty(self.num_nodes))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.thresholds, self.init_threshold)

    def forward(self, edge_index, confidence, base_edge_weight=None):
        source = edge_index[0]
        target = edge_index[1]
        confidence = confidence.to(device=self.thresholds.device, dtype=self.thresholds.dtype).view(-1)
        if confidence.numel() != self.num_nodes:
            raise ValueError(f"Expected confidence for {self.num_nodes} nodes, got {confidence.numel()}.")

        if base_edge_weight is None:
            base = torch.ones(source.numel(), device=self.thresholds.device, dtype=self.thresholds.dtype)
        else:
            base = base_edge_weight.to(device=self.thresholds.device, dtype=self.thresholds.dtype).view(-1)
            if base.numel() != source.numel():
                raise ValueError("base_edge_weight must have one value per edge.")

        raw = torch.sigmoid(confidence[source] - self.thresholds[target]) / 0.5
        beta = torch.full_like(raw, self.beta)
        # Forward values match the UnGSL rule exactly; the low branch uses a
        # straight-through gradient so node thresholds can recover from a
        # conservative initialization instead of becoming permanently frozen.
        gate = torch.where(raw >= 1.0, raw, beta + raw - raw.detach())
        return base * gate


UComFyGate = UncertaintyEdgeGate
