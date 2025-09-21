# Python
import os, sys, subprocess
from pathlib import Path

# Get the project root (where this script is run from)
PROJECT_ROOT = Path.cwd()
REPO_DIR = PROJECT_ROOT / "mnet_repro"
MANIFEST = PROJECT_ROOT / "data" / "PROMISE12" / "preprocessed" / "dataset_manifest.json"
OUTDIR = PROJECT_ROOT / "outputs" / "debug_run"

# Change to the mnet_repro directory to run train.py
os.chdir(REPO_DIR)

# Use absolute paths for the manifest and output directory
cmd = [
    sys.executable, "train.py",
    "--config", "mnet/configs/promis12_mnet.yaml",
    "--manifest", str(MANIFEST),
    "--epochs", "2",
    "--bs", "2",
    "--patchD", "16", "--patchH", "128", "--patchW", "128",
    "--strideD", "8", "--strideH", "64", "--strideW", "64",
    "--outdir", str(OUTDIR),
    "--amp",   # keep AMP on for speed
]
print(">>>", " ".join(cmd))
subprocess.check_call(cmd)

print("\nArtifacts:")
print("  best.pt  :", (OUTDIR / "best.pt").exists())
print("  last.pt  :", (OUTDIR / "last.pt").exists())
print("  train_log:", (OUTDIR / "train_log.jsonl").exists())

# 6b) (Optional) SGD + poly LR quick run
OUT2 = PROJECT_ROOT / "outputs/mnet_paperish_sgdpoly"
cmd2 = cmd[:-2] + ["--outdir", str(OUT2), "--sgd_poly"]
print("\n>>>", " ".join(cmd2))
subprocess.check_call(cmd2)

print("✅ Paper-aligned training passes executed.")
print("\nArtifacts:")
print("  best.pt  :", (OUT2 / "best.pt").exists())
print("  last.pt  :", (OUT2 / "last.pt").exists())
print("  train_log:", (OUT2 / "train_log.jsonl").exists())




