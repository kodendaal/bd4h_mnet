import os, argparse, json, time, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import SimpleITK as sitk


# Add the current directory to Python path so we can import mnet
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from mnet.utils.config import load_yaml_with_env
from mnet.utils.seed import set_seed
from mnet.losses.combined import DiceCELoss
from mnet.utils.dice import dice_mean, dice_per_class
from mnet.utils.aggregator import VolumeLogitAggregator
from mnet.utils.train_utils import AvgMeter, TrainState, save_ckpt
from mnet.data.dataset_promise12 import Promise12Patches
from mnet.models.mnet_tiny import MNetTiny
from mnet.models.mnet import MNet
from mnet.models.mnet_officialish import MNetOfficialish

def numpy_collate(batch):
    outs = {}
    for k in batch[0].keys():
        vals = [b[k] for b in batch]
        if k in ("image","label") and vals[0] is not None:
            outs[k] = torch.from_numpy(np.stack(vals, axis=0))
        else:
            outs[k] = vals
    return outs

def build_loaders(manifest, patch=(16,128,128), stride=(8,64,64), bs=2, nw=2):
    train_ds = Promise12Patches(manifest_path=manifest, split="train",
                                patch_DHW=patch, mode="random",
                                aug={"intensity_p":0.2,"flip_p":0.5})
    val_ds   = Promise12Patches(manifest_path=manifest, split="val",
                                patch_DHW=patch, mode="sliding", stride_DHW=stride)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True, collate_fn=numpy_collate)
    val_dl   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=0,  pin_memory=False, collate_fn=numpy_collate)
    return train_dl, val_dl

def validate_full_volume(val_dl, model, device, num_classes=2):
    model.eval()
    dices, dices_bg, dices_fg = [], [], []
    case_count = 0
    total_batches = len(val_dl)
    
    with torch.no_grad():
        cur_case = None
        aggregator = None
        gt_volume = None  # torch LongTensor on CPU: [1, D, H, W]
        
        for batch_idx, batch in enumerate(val_dl):
            imgs   = batch["image"].to(device)          # [1,1,d,h,w]
            case   = batch["case"][0]
            start  = tuple(map(int, batch["start"][0]))  # (z,y,x)
            valid  = tuple(map(int, batch["valid"][0]))  # (dv,hv,wv)
            D,H,W  = map(int, batch["img_shape"][0])     # full canvas size
            lblpth = batch.get("label_path", [""])[0]

            # New case: finalize previous, then init new aggregator and load full GT
            if case != cur_case:
                if aggregator is not None and gt_volume is not None:
                    avg_logits = aggregator.get_avg().unsqueeze(0)  # [1,C,D,H,W]
                    d   = dice_mean(avg_logits, gt_volume.to(device), exclude_background=False).item()
                    dpc = dice_per_class(avg_logits, gt_volume.to(device)).cpu().numpy()
                    dices.append(d); dices_bg.append(dpc[0]); dices_fg.append(dpc[1] if len(dpc)>1 else dpc[0])
                    print(f"     ✅ Case {cur_case} completed (Dice: {d:.4f})")

                cur_case = case
                case_count += 1
                aggregator = VolumeLogitAggregator((D,H,W), num_classes=num_classes, device=device)

                # Load full GT label once (if available); shape [1, D, H, W] long
                if lblpth and Path(lblpth).exists():
                    gt_img = sitk.ReadImage(str(lblpth))
                    gt_arr = sitk.GetArrayFromImage(gt_img).astype(np.int64)  # [D,H,W]
                    gt_volume = torch.from_numpy(gt_arr)[None, ...]           # [1,D,H,W]
                    print(f"     🔄 Processing case {case_count}: {case} (volume: {D}x{H}x{W}) - GT loaded")
                else:
                    gt_volume = None  # if no GT for test split
                    print(f"     🔄 Processing case {case_count}: {case} (volume: {D}x{H}x{W}) - No GT available")

            # forward current patch
            out = model(imgs)

            if isinstance(out, list):
                logits = out[0]     # X55 (main) — same as official
            elif isinstance(out, dict):
                logits = out["main"]
            else:
                logits = out
            # logits = out["main"] if isinstance(out, dict) else out  # [1,C,d,h,w]
            aggregator.add(logits[0], start, valid)
            
            # Show progress every 5 batches
            if (batch_idx + 1) % 5 == 0:
                print(f"     📊 Batch {batch_idx + 1:3d}/{total_batches:3d} | Case: {case}")

        # finalize last case
        if aggregator is not None and gt_volume is not None:
            avg_logits = aggregator.get_avg().unsqueeze(0)
            d   = dice_mean(avg_logits, gt_volume.to(device), exclude_background=False).item()
            dpc = dice_per_class(avg_logits, gt_volume.to(device)).cpu().numpy()
            dices.append(d); dices_bg.append(dpc[0]); dices_fg.append(dpc[1] if len(dpc)>1 else dpc[0])
            print(f"     ✅ Case {cur_case} completed (Dice: {d:.4f})")

    model.train()
    
    # Calculate final statistics
    dice_mean_val = float(np.mean(dices)) if dices else 0.0
    dice_bg_val = float(np.mean(dices_bg)) if dices_bg else 0.0
    dice_fg_val = float(np.mean(dices_fg)) if dices_fg else 0.0
    
    print(f"     📈 Validation summary:")
    print(f"        - Cases processed: {len(dices)}")
    print(f"        - Mean Dice: {dice_mean_val:.4f}")
    print(f"        - Background Dice: {dice_bg_val:.4f}")
    print(f"        - Foreground Dice: {dice_fg_val:.4f}")
    
    return {
        "dice_mean": dice_mean_val,
        "dice_bg":   dice_bg_val,
        "dice_fg":   dice_fg_val,
        "n_cases":   len(dices)
    }

def build_model(cfg):

    mcfg = cfg.get("model", {})
    name = mcfg.get("name", "MNet")
    in_ch = int(mcfg.get("in_channels", 1))
    out_ch = int(mcfg.get("out_channels", 2))

    if name == "MNetOfficialish":
        kn  = tuple(mcfg.get("kn", [32,48,64,80,96]))
        ds  = bool(mcfg.get("ds", True))
        fmu = str(mcfg.get("fmu", "sub"))
        return MNetOfficialish(in_channels=in_ch, num_classes=out_ch, kn=kn, ds=ds, FMU=fmu)

    base_ch = int(mcfg.get("base_ch", 32)) 
    depth = int(mcfg.get("depth", 4))
    deep_supervision = bool(mcfg.get("deep_supervision", True))
    ds_heads = int(mcfg.get("ds_heads", 3))
    if name == "MNet":
        return MNet(in_channels=in_ch, out_channels=out_ch, base_ch=base_ch, depth=depth,
                    deep_supervision=deep_supervision, ds_heads=ds_heads)
    return MNetTiny(in_channels=in_ch, out_channels=out_ch)

def poly_lr(it, it_max, lr0, power=0.9):
    return lr0 * ((1 - float(it)/float(it_max))**power)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="mnet/configs/promis12_mnet.yaml")
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--patchD", type=int, default=16)
    ap.add_argument("--patchH", type=int, default=128)
    ap.add_argument("--patchW", type=int, default=128)
    ap.add_argument("--strideD", type=int, default=8)
    ap.add_argument("--strideH", type=int, default=64)
    ap.add_argument("--strideW", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--outdir", type=str, default="checkpoints")
    ap.add_argument("--sgd_poly", action="store_true", default=False)
    args, _ = ap.parse_known_args()

    print("🚀 Starting training script...")
    print(f"📁 Manifest: {args.manifest}")
    print(f"📁 Output directory: {args.outdir}")
    
    cfg = load_yaml_with_env(args.config)
    set_seed(cfg.get("seed", 42))
    print(f"🎲 Seed set to: {cfg.get('seed', 42)}")

    epochs = args.epochs if args.epochs is not None else int(cfg["train"].get("epochs", 20))
    lr0    = args.lr     if args.lr     is not None else float(cfg["train"].get("lr", 1e-3))
    patch  = (args.patchD, args.patchH, args.patchW)
    stride = (args.strideD, args.strideH, args.strideW)
    
    print(f"⚙️  Training config:")
    print(f"   - Epochs: {epochs}")
    print(f"   - Learning rate: {lr0}")
    print(f"   - Patch size: {patch}")
    print(f"   - Stride: {stride}")
    print(f"   - Batch size: {args.bs}")
    print(f"   - AMP enabled: {args.amp}")
    print(f"   - SGD Poly: {args.sgd_poly}")

    model = build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"🤖 Model created: {cfg.get('model', {}).get('name', 'MNet')}")
    print(f"   - Input channels: {int(cfg['model']['in_channels'])}")
    print(f"   - Output channels: {int(cfg['model']['out_channels'])}")
    print(f"   - Device: {device}")

    print("📊 Loading datasets...")
    train_dl, val_dl = build_loaders(args.manifest, patch=patch, stride=stride, bs=args.bs, nw=2)
    print(f"   - Training batches: {len(train_dl)}")
    print(f"   - Validation batches: {len(val_dl)}")

    crit = DiceCELoss()
    if args.sgd_poly:
        optim = torch.optim.SGD(model.parameters(), lr=lr0, momentum=0.99, nesterov=False)
        it_max = max(1, epochs * len(train_dl))
        print(f"🔧 Optimizer: SGD (lr={lr0}, momentum=0.99)")
    else:
        optim = torch.optim.Adam(model.parameters(), lr=lr0)
        it_max = None
        print(f"🔧 Optimizer: Adam (lr={lr0})")
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type=="cuda")
    print(f"📉 Loss: DiceCE")
    print(f"⚡ AMP: {'Enabled' if args.amp and device.type=='cuda' else 'Disabled'}")

    state = TrainState(epoch=0, global_step=0, best_val=-1.0)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Checkpoints will be saved to: {outdir}")
    print("=" * 60)
    print("🏃 Starting training loop...")
    print("=" * 60)

    for epoch in range(1, epochs+1):
        print(f"\n📅 Epoch {epoch}/{epochs}")
        loss_meter = AvgMeter()
        batch_count = 0
        t0 = time.time()
        
        for batch in train_dl:
            batch_count += 1
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp and device.type=="cuda"):
                preds = model(x)
                # inside training step, after preds = model(x)
                if isinstance(preds, list):
                    # Paper Eq.(5): main + 6 aux with geometric decay.
                    # We don't have exact Xi5/X5i mapping here, but the official order is:
                    # [X55 (main), X5?, X?5, X?2, ...]  We apply weights by depth (coarse->fine):
                    weights = [1.0, 0.5, 0.5, 0.25, 0.25, 0.125, 0.125]  # matches λ_i=(1/2)^{5-i} pairs
                    loss = 0.0
                    for i, head in enumerate(preds):
                        Dy,Hy,Wy = head.shape[2:]
                        # downsample labels to head resolution (nearest)
                        y_i = F.interpolate(y.unsqueeze(1).float(), size=(Dy,Hy,Wy), mode="nearest").squeeze(1).long()
                        loss = loss + weights[i] * crit(head, y_i)
                elif isinstance(preds, dict):
                    # your existing deep-supervision dict path
                    # main
                    loss = crit(preds["main"], y)
                    # aux: coarse->fine; λ_i = (1/2)^(K-1-i)
                    Ks = len(preds["aux"])
                    for i, a in enumerate(preds["aux"]):
                        Dy,Hy,Wy = a.shape[2:]
                        y_i = F.interpolate(y.unsqueeze(1).float(), size=(Dy,Hy,Wy), mode="nearest").squeeze(1).long()
                        lam = 0.5 ** (Ks-1-i)
                        loss = loss + lam * crit(a, y_i)
                else:
                    loss = crit(preds, y)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            if args.sgd_poly:
                state.global_step += 1
                new_lr = poly_lr(state.global_step, it_max, lr0)
                for g in optim.param_groups: g["lr"] = new_lr
            loss_meter.update(loss.item(), k=x.size(0))
            
            # Print progress every 10 batches
            if batch_count % 5 == 0:
                current_lr = optim.param_groups[0]['lr']
                print(f"   Batch {batch_count:3d}/{len(train_dl):3d} | Loss: {loss.item():.4f} | Avg Loss: {loss_meter.avg:.4f} | LR: {current_lr:.6f}")

        print(f"   🔍 Running validation...")
        val_stats = validate_full_volume(val_dl, model, device, num_classes=int(cfg["model"]["out_channels"]))
        elapsed = time.time() - t0
        
        print(f"✅ Epoch {epoch:03d} completed!")
        print(f"   📊 Training loss: {loss_meter.avg:.4f}")
        print(f"   🎯 Validation Dice: {val_stats['dice_mean']:.4f}")
        print(f"   📈 Background Dice: {val_stats['dice_bg']:.4f}")
        print(f"   📈 Foreground Dice: {val_stats['dice_fg']:.4f}")
        print(f"   ⏱️  Time: {elapsed:.1f}s")

        is_best = val_stats["dice_mean"] > state.best_val
        if is_best:
            state.best_val = val_stats["dice_mean"]
            print(f"   🏆 New best model! Saving to best.pt (Dice: {val_stats['dice_mean']:.4f})")
            save_ckpt(outdir / "best.pt", model, optim, scaler, state, extra={"val": val_stats})
        else:
            print(f"   💾 Saving checkpoint to last.pt")
        save_ckpt(outdir / "last.pt", model, optim, scaler, state, extra={"val": val_stats})
        
        state.epoch = epoch

    print("=" * 60)
    print("🎉 Training completed!")
    print(f"🏆 Best validation Dice: {state.best_val:.4f}")
    print(f"💾 Final checkpoints saved to: {outdir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
