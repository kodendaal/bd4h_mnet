import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import os

# Add the parent directory to Python path so we can import mnet
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from mnet.data.dataset_promise12 import Promise12Patches

def numpy_collate(batch):
    # Convert numpy arrays to torch here (keeps Dataset torch-free)
    outs = {}
    for key in batch[0].keys():
        vals = [b[key] for b in batch]
        if key in ("image","label") and vals[0] is not None:
            # stack along batch
            if key == "image":
                outs[key] = torch.from_numpy(np.stack(vals, axis=0))
            else:
                outs[key] = torch.from_numpy(np.stack(vals, axis=0))
        else:
            outs[key] = vals
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--patchD", type=int, default=16)
    ap.add_argument("--patchH", type=int, default=128)
    ap.add_argument("--patchW", type=int, default=128)
    ap.add_argument("--mode", type=str, default="random", choices=["random","sliding"])
    ap.add_argument("--strideD", type=int, default=8)
    ap.add_argument("--strideH", type=int, default=64)
    ap.add_argument("--strideW", type=int, default=64)
    ap.add_argument("--bs", type=int, default=2)
    args = ap.parse_args()

    ds = Promise12Patches(
        manifest_path=Path(args.manifest),
        split=args.split,
        patch_DHW=(args.patchD, args.patchH, args.patchW),
        mode=args.mode,
        stride_DHW=(args.strideD, args.strideH, args.strideW),
        aug={"intensity_p": 0.2, "flip_p": 0.5} if args.split=="train" else {}
    )
    dl = DataLoader(ds, batch_size=args.bs, shuffle=(args.mode=="random"), num_workers=2, collate_fn=numpy_collate)

    it = iter(dl)
    batch = next(it)
    x = batch["image"]     # [B, 1, D, H, W]
    y = batch["label"]     # [B, D, H, W] or None for test
    cases = batch["case"]
    print("Batch image:", tuple(x.shape), x.dtype)
    print("Batch label:", None if y is None else tuple(y.shape), None if y is None else y.dtype)
    print("Cases:", cases[:4])

    # quick forward through our tiny model to verify pipeline end-to-end
    from mnet.models.mnet_tiny import MNetTiny
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MNetTiny(in_channels=1, out_channels=2).to(device).eval()
    with torch.no_grad():
        yhat = net(x.to(device))
    print("MNetTiny logits shape:", tuple(yhat.shape))

if __name__ == "__main__":
    main()
