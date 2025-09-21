import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        # logits: [B, C, ...]; targets: [B, ...] (long) or one-hot [B, C, ...]
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        if targets.dtype == torch.long:
            # make one-hot
            one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes)
            # move channel to dim=1
            one_hot = one_hot.permute(0, -1, *range(1, targets.ndim)).contiguous()
        else:
            one_hot = targets
        one_hot = one_hot.type_as(probs)

        dims = tuple(range(2, probs.ndim))
        intersection = torch.sum(probs * one_hot, dims)
        cardinality = torch.sum(probs + one_hot, dims)
        dice = (2. * intersection + self.eps) / (cardinality + self.eps)
        return 1. - dice.mean()
