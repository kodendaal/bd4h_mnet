import torch.nn as nn
from .dice import DiceLoss

class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.dw = dice_weight
        self.cw = ce_weight

    def forward(self, logits, targets):
        return self.dw * self.dice(logits, targets) + self.cw * self.ce(logits, targets)
