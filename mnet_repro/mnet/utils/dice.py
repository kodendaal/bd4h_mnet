import numpy as np
import torch
import torch.nn.functional as F

def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    # labels: [B, D, H, W] (int64). Returns [B, C, D, H, W] (float32)
    return F.one_hot(labels.long(), num_classes=num_classes).permute(0,4,1,2,3).float()

@torch.no_grad()
def dice_per_class(logits: torch.Tensor, labels: torch.Tensor, eps: float = 1e-6):
    """
    logits: [B, C, D, H, W] (float)
    labels: [B, D, H, W] (long)
    returns: per-class dice, shape [C] (torch.float)
    """
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    target = one_hot(labels, num_classes=num_classes).type_as(probs)
    dims = tuple(range(2, probs.ndim))
    inter = torch.sum(probs * target, dims)
    denom = torch.sum(probs + target, dims)
    dice = (2 * inter + eps) / (denom + eps)  # [B, C]
    return dice.mean(dim=0)  # [C]

@torch.no_grad()
def dice_mean(logits: torch.Tensor, labels: torch.Tensor, exclude_background: bool = False) -> torch.Tensor:
    dpc = dice_per_class(logits, labels)  # [C]
    if exclude_background and dpc.numel() > 1:
        return dpc[1:].mean()
    return dpc.mean()
