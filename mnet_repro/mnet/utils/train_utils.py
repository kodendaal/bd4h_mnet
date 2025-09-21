import torch
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict

class AvgMeter:
    def __init__(self):
        self.n = 0
        self.sum = 0.0
    def update(self, val, k=1):
        self.sum += float(val) * k
        self.n += k
    @property
    def avg(self):
        return self.sum / max(1, self.n)

@dataclass
class TrainState:
    epoch: int
    global_step: int
    best_val: float

def save_ckpt(path: Path, model, optim, scaler, state: TrainState, extra: Dict[str, Any] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict() if optim else None,
        "scaler": scaler.state_dict() if scaler else None,
        "state": asdict(state),
        "extra": extra or {}
    }, str(path))

def load_ckpt(path: Path, model, optim=None, scaler=None):
    obj = torch.load(str(path), map_location="cpu")
    model.load_state_dict(obj["model"], strict=True)
    if optim and obj.get("optim") is not None:
        optim.load_state_dict(obj["optim"])
    if scaler and obj.get("scaler") is not None:
        scaler.load_state_dict(obj["scaler"])
    return obj
