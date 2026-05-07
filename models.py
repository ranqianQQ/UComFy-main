from typing import Iterable

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from modules.ucomfy_gate import UComFyGate


class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index, edge_weight=edge_weight)


class UComFyGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_nodes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        beta: float = 0.2,
        init_threshold: float = 0.5,
    ):
        super().__init__()
        self.gcn = GCN(in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout)
        self.gate = UComFyGate(num_nodes=num_nodes, beta=beta, init_threshold=init_threshold)

    @property
    def thresholds(self):
        return self.gate.thresholds

    def reset_parameters(self) -> None:
        self.gcn.reset_parameters()
        self.gate.reset_parameters()

    def ungsl_parameters(self) -> Iterable[nn.Parameter]:
        return [self.gate.thresholds]

    def gcn_parameters(self) -> Iterable[nn.Parameter]:
        threshold_id = id(self.gate.thresholds)
        return [param for param in self.parameters() if id(param) != threshold_id]

    def forward(self, x, edge_index, confidence=None, base_edge_weight=None):
        edge_weight = base_edge_weight
        if confidence is not None:
            edge_weight = self.gate(edge_index, confidence, base_edge_weight)
        return self.gcn(x, edge_index, edge_weight=edge_weight)
