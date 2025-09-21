import os
from pathlib import Path
import os, sys, subprocess, traceback
import textwrap

# You can change these if you want a different folder structure:
REPO_DIR = Path("./mnet_repro")
DATA_ROOT = Path("./data")        # we'll use this later (Step 2)
OUTPUT_ROOT = Path("./outputs")   # logs/checkpoints later

REPO_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("Repo dir:", REPO_DIR)
print("Data root:", DATA_ROOT)
print("Output root:", OUTPUT_ROOT)


def create_subdirs():
    subdirs = [
        "mnet/configs",
        "mnet/data",
        "mnet/losses",
        "mnet/models",
        "mnet/utils",
        "scripts",
        "env",
        "tests",
    ]

    for sd in subdirs:
        (REPO_DIR / sd).mkdir(parents=True, exist_ok=True)

    print("Created subdirectories under", REPO_DIR)


def wfile(path: Path, content: str):
    path.write_text(textwrap.dedent(content).lstrip())


def populate_subdirs():

    # --- README.md ---
    wfile(REPO_DIR / "README.md", """
        # MNet Reproduction (PyTorch, unique implementation)

        This repository aims to *faithfully reproduce* the core functionality of
        "MNet: Rethinking 2D/3D Networks for Anisotropic Medical Image Segmentation"
        in a clean-room PyTorch codebase, and then extend it with:
        1) Oblique data transformations
        2) 2D–3D gate fusion

        ## Structure
        - mnet/
        - configs/           # YAML experiment configs (paths, hyperparams)
        - data/              # Dataset loaders, transforms, preprocessing scripts
        - losses/            # Dice, CE, combined, etc.
        - models/            # MNet and baseline architectures
        - utils/             # I/O, logging, metrics, helpers
        - scripts/             # CLI helpers for data prep, training, evaluation
        - env/                 # Requirements and environment files
        - tests/               # Lightweight tests and smoke checks

        ## Paths
        - Paths are configurable via CLI and/or environment variables (see configs).
        - Example (Colab): DATA_ROOT=/content/data, OUTPUT_ROOT=/content/outputs
    """)

    # --- .gitignore ---
    wfile(REPO_DIR / ".gitignore", """
        __pycache__/
        *.pyc
        *.pyo
        *.pyd
        .build/
        dist/
        .env
        .venv
        *.egg-info/
        .DS_Store
        .ipynb_checkpoints/
        outputs/
        checkpoints/
        wandb/
    """)

    # --- Minimal config (YAML) ---
    wfile(REPO_DIR / "mnet/configs/promis12_example.yaml", """
        experiment_name: "mnet_promis12_debug"
        seed: 42

        paths:
        # You can override these via env vars or CLI flags in train.py
        data_root: "${DATA_ROOT:/content/data}"
        output_root: "${OUTPUT_ROOT:/content/outputs}"

        data:
        dataset: "PROMISE12"
        fold: 0
        train_split: "train"
        val_split: "val"
        test_split: "test"

        train:
        epochs: 2
        batch_size: 1
        lr: 1e-3
        amp: true

        model:
        name: "MNetTiny"
        in_channels: 1
        out_channels: 2
    """)

    # --- utils/config.py ---
    wfile(REPO_DIR / "mnet/utils/config.py", r"""
        import os, re, yaml

        _env_pat = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\:([^}]*)\}")

        def _expand_env_defaults(s: str) -> str:
            def repl(m):
                key, default = m.group(1), m.group(2)
                return os.environ.get(key, default)
            return _env_pat.sub(repl, s)

        def load_yaml_with_env(path: str):
            with open(path, "r") as f:
                raw = f.read()
            raw = _expand_env_defaults(raw)
            return yaml.safe_load(raw)
    """)

    # --- utils/seed.py ---
    wfile(REPO_DIR / "mnet/utils/seed.py", """
        import random, numpy as np, torch

        def set_seed(seed: int = 42):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = False  # allow perf
            torch.backends.cudnn.benchmark = True
    """)

    # --- losses/dice.py ---
    wfile(REPO_DIR / "mnet/losses/dice.py", """
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
    """)

    # --- losses/combined.py ---
    wfile(REPO_DIR / "mnet/losses/combined.py", """
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
    """)

    # --- models/blocks.py ---
    wfile(REPO_DIR / "mnet/models/blocks.py", """
        import torch.nn as nn

        def conv_norm_act(in_ch, out_ch, k=3, s=1, p=1, dim=3):
            if dim == 3:
                conv = nn.Conv3d(in_ch, out_ch, k, s, p, bias=False)
                norm = nn.InstanceNorm3d(out_ch, affine=True)
            elif dim == 2:
                conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
                norm = nn.InstanceNorm2d(out_ch, affine=True)
            else:
                raise ValueError("dim must be 2 or 3")
            return nn.Sequential(conv, norm, nn.ReLU(inplace=True))
    """)

    # --- models/mnet_tiny.py ---
    wfile(REPO_DIR / "mnet/models/mnet_tiny.py", """
        import torch.nn as nn
        from .blocks import conv_norm_act

        class MNetTiny(nn.Module):
            \"\"\"Minimal placeholder network so that we can validate the pipeline.
            This is NOT the final MNet — it will be replaced by the faithful implementation.
            \"\"\"
            def __init__(self, in_channels=1, out_channels=2):
                super().__init__()
                self.enc = nn.Sequential(
                    conv_norm_act(in_channels, 16, dim=3),
                    conv_norm_act(16, 32, dim=3),
                )
                self.head = nn.Conv3d(32, out_channels, kernel_size=1)

            def forward(self, x):  # x: [B, C, D, H, W]
                z = self.enc(x)
                return self.head(z)
    """)

    # --- train.py (smoke test only for now) ---
    wfile(REPO_DIR / "train.py", """
        import os, argparse, torch
        from mnet.utils.config import load_yaml_with_env
        from mnet.utils.seed import set_seed
        from mnet.models.mnet_tiny import MNetTiny
        from mnet.losses.combined import DiceCELoss

        def get_args():
            ap = argparse.ArgumentParser()
            ap.add_argument("--config", type=str, default="mnet/configs/promis12_example.yaml")
            ap.add_argument("--data_root", type=str, default=None)
            ap.add_argument("--output_root", type=str, default=None)
            return ap.parse_args()

        def main():
            args = get_args()
            cfg = load_yaml_with_env(args.config)
            set_seed(cfg.get("seed", 42))

            # Allow CLI overrides of paths:
            if args.data_root:  cfg["paths"]["data_root"]  = args.data_root
            if args.output_root: cfg["paths"]["output_root"] = args.output_root

            # Model (placeholder)
            model = MNetTiny(
                in_channels=cfg["model"]["in_channels"],
                out_channels=cfg["model"]["out_channels"]
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device).train()

            # Dummy tensors (no real data yet)
            x = torch.randn(1, cfg["model"]["in_channels"], 16, 64, 64, device=device)  # [B,C,D,H,W]
            y = torch.zeros(1, 16, 64, 64, dtype=torch.long, device=device)              # [B,D,H,W] (class 0)

            crit = DiceCELoss()
            optim = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

            for _ in range(2):
                optim.zero_grad(set_to_none=True)
                logits = model(x)
                loss = crit(logits, y)
                loss.backward()
                optim.step()

            print("Smoke training OK. Final loss:", float(loss.detach().cpu()))

        if __name__ == "__main__":
            main()
    """)

    # --- evaluate.py (placeholder) ---
    wfile(REPO_DIR / "evaluate.py", """
        # Placeholder for evaluation entrypoint; implemented in later steps.
        print("Evaluation placeholder.")
    """)

    # --- tests/test_smoke.py (optional) ---
    wfile(REPO_DIR / "tests/test_smoke.py", """
        import torch
        from mnet.models.mnet_tiny import MNetTiny

        def test_forward():
            net = MNetTiny(1, 2).eval()
            x = torch.randn(1,1,16,64,64)
            with torch.no_grad():
                y = net(x)
            assert y.shape == (1,2,16,64,64)
    """)

    print("Files written to", REPO_DIR)

def create_inits():
        
    packages = [
        REPO_DIR / "mnet",
        REPO_DIR / "mnet/configs",
        REPO_DIR / "mnet/data",
        REPO_DIR / "mnet/losses",
        REPO_DIR / "mnet/models",
        REPO_DIR / "mnet/utils",
        REPO_DIR / "scripts",
        REPO_DIR / "tests",
    ]
    for p in packages:
        p.mkdir(parents=True, exist_ok=True)
        init = p / "__init__.py"
        if not init.exists():
            init.write_text("")  # make it a package

    print("Added __init__.py to all packages.")


def pipeline_smoke():

    os.chdir(REPO_DIR)
    print("CWD:", os.getcwd())

    # Ensure env vars so config expands paths
    os.environ["DATA_ROOT"] = "./data"
    os.environ["OUTPUT_ROOT"] = "./outputs"

    # Run directly so you can see full Python traceback in the cell output
    try:
        import runpy
        runpy.run_path("train.py", run_name="__main__")
        print("✅ Smoke training ran successfully.")
    except SystemExit as e:
        print("train.py exited with code:", e.code)
        raise
    except Exception as e:
        print("❌ Exception while running train.py\n")
        traceback.print_exc()
        raise


def main():
    create_subdirs()
    populate_subdirs()
    create_inits()
    pipeline_smoke()


if __name__ == "__main__":
    main()
