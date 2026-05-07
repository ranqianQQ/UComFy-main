import hashlib
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def entropy_from_logits(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + eps)).sum(dim=-1)


def confidence_from_entropy(entropy: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    confidence = torch.exp(-entropy)
    if normalize:
        min_val = confidence.min()
        max_val = confidence.max()
        confidence = (confidence - min_val) / (max_val - min_val + 1e-12)
    return confidence


def confidence_from_logits(logits: torch.Tensor, normalize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    entropy = entropy_from_logits(logits)
    confidence = confidence_from_entropy(entropy, normalize=normalize)
    return entropy, confidence


def confidence_cache_path(
    cache_dir: str,
    dataset: str,
    seed: int,
    split: int,
    rewire_method: str,
    add_budget: int,
    delete_budget: int,
    normalize: bool,
) -> str:
    raw = f"{dataset}|{seed}|{split}|{rewire_method}|{add_budget}|{delete_budget}|{int(normalize)}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    safe_dataset = dataset.replace(".", "_").replace("/", "_").replace("\\", "_")
    filename = f"{safe_dataset}_{rewire_method}_seed{seed}_split{split}_{digest}.pt"
    return os.path.join(cache_dir, filename)


def load_cached_confidence(path: str, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not os.path.exists(path):
        return None, None
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    return payload.get("entropy"), payload.get("confidence")


def save_cached_confidence(path: str, entropy: torch.Tensor, confidence: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"entropy": entropy.detach().cpu(), "confidence": confidence.detach().cpu()}, path)
