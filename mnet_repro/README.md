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
