# Reproducing MNet on PROMISE12

## 0) Prerequisites

* Python 3.8–3.10 with CUDA-enabled PyTorch.
* nnU-Net v1 (this repo includes our trainer + model under `nnunet/.../reproduction_mnet`).
* PROMISE12 dataset downloaded.

> If you’re on Windows, strongly prefer WSL2/Ubuntu. If you must use native Windows, make sure paths are short and not under OneDrive.

Our custom reproduction files are found in the following locations:

```text
<repo-root>/
├─ nnUNet/                                # nnU-Net v1 checkout (installed with `pip install -e`)
   ├─ nnunet/
   │  ├─ dataset_conversion/
   │  │  └─ Task024_Promise2012.py        # PROMISE → nnU-Net layout converter
   │  ├─ experiment_planning/
   │  │  ├─ nnUNet_plan_and_preprocess.py
   │  │  └─ alternative_experiment_planning/
   │  │     └─ target_spacing/
   │  │        ├─ ExperimentPlanner3D_v21_trgSp_z1.0_yx0.6125.py
   │  │        ├─ ExperimentPlanner3D_v21_trgSp_z2.2_yx0.6125.py
   │  │        └─ ExperimentPlanner3D_v21_trgSp_z4.0_yx0.6125.py
   │  ├─ training/
   │  │  └─ network_training/
   │  │     └─ myTrainer_reproduction.py  # ← your custom trainer (builds your MNet)
   │  └─ network_architecture/
   │     └─ reproduction_mnet/
   │        ├─ __init__.py
   │        ├─ basic_models.py            # ← your blocks/backbones
   │        └─ mnet.py                    # ← your MNet model used by the trainer
   ├─ setup.py
   ├─ setup.cfg
   ├─ LICENSE
   └─ README.md
```

---

## 1) Environment setup

```bash
# create/activate env (example)
conda create -n mnet-repro python=3.10 -y
conda activate mnet-repro

# install repo (editable)
pip install -e third_party/nnUNet   # path to your nnUNet v1 checkout
pip install -e .                    # your top-level repo if needed (for myTrainer_reproduction import path)

# set nnU-Net required env vars FOR THIS SHELL
export nnUNet_raw_data_base=/teamspace/studios/this_studio/nnUNet_raw
export nnUNet_preprocessed=/teamspace/studios/this_studio/nnUNet_preprocessed
export RESULTS_FOLDER=/teamspace/studios/this_studio/nnUNet_results

# pick GPU
export CUDA_VISIBLE_DEVICES=0
```

Create the directories if they don’t exist:

```bash
mkdir -p "$nnUNet_raw_data_base/nnUNet_raw_data" \
         "$nnUNet_preprocessed" \
         "$RESULTS_FOLDER"
```

---

## 2) Convert PROMISE12 to nnU-Net layout

Put PROMISE12 original files under (for example) `/teamspace/data/PROMISE12/{train,test}` as `.mhd` & segmentations. Then:

```bash
python -m nnunet.dataset_conversion.Task024_Promise2012 \
  --source /teamspace/data/PROMISE12 \
  --target "$nnUNet_raw_data_base/nnUNet_raw_data/Task024_Promise"
```

This creates:

```
nnUNet_raw/nnUNet_raw_data/Task024_Promise/
  imagesTr/CaseXX_0000.nii.gz
  labelsTr/CaseXX.nii.gz
  imagesTs/CaseYY_0000.nii.gz
  dataset.json
```

> If you prefer, you can run the original script in that module without args and edit the top paths, but the command above is easiest if your fork supports args.

---

## 3) Baseline planning & preprocessing (default spacing)

This gives you a known-good nnU-Net baseline (useful sanity check).

```bash
nnUNet_plan_and_preprocess -t 24 -planner3d ExperimentPlanner3D_v21 --verify_dataset_integrity
```

Artifacts written to:

```
$nnUNet_preprocessed/Task024_Promise/nnUNetPlansv2.1_plans_3D.pkl
$nnUNet_preprocessed/Task024_Promise/nnUNetData_plans_v2.1/...
```

---

## 4) Target-spacing variants (z sweep)

nnU-Net v1 already includes examples in
`nnunet/experiment_planning/alternative_experiment_planning/target_spacing/`:

* `experiment_planner_baseline_3DUNet_v21_customTargetSpacing_2x2x2.py` (returns `[2.,2.,2.]`)
* `experiment_planner_baseline_3DUNet_targetSpacingForAnisoAxis.py` (auto-tighten along the worst axis)

You can clone the 2×2×2 file into **one file per z** and change the return line, e.g.:

```python
# Example: ExperimentPlanner3D_v21_trgSp_z2p2_yx0p6125
def get_target_spacing(self):
    import numpy as np
    return np.array([2.2, 0.6125, 0.6125])
```

Name each class uniquely; each sets its own `self.data_identifier` and `self.plans_fname` inside `__init__` so the **plans IDs will be unique** and won’t overwrite each other.

Then plan & preprocess for each spacing:

```bash
# z = 1.0
nnUNet_plan_and_preprocess -t 24 -pl3d ExperimentPlanner3D_v21_trgSp_z1p0_yx0p6125 --verify_dataset_integrity 

# z = 2.2
nnUNet_plan_and_preprocess -t 24 -pl3d ExperimentPlanner3D_v21_trgSp_z2p2_yx0p6125 --verify_dataset_integrity

# z = 4.0
nnUNet_plan_and_preprocess -t 24 -pl3d ExperimentPlanner3D_v21_trgSp_z4p0_yx0p6125 --verify_dataset_integrity
```

After each run, check you have a **`*_plans_3D.pkl`** file per variant in:

```
$nnUNet_preprocessed/Task024_Promise/
  <YourPlansName>_plans_3D.pkl
```

The **plans identifier** passed to `-p` during training is the filename **without** `_plans_3D.pkl`.

> Use decimal points in filenames (e.g., `z1.0`, not `z1p0`) unless you consistently used `p` everywhere.

---

## 5) Training

### 5.1 Single run

```bash
# example: z=2.2 plan
nnUNet_train 3d_fullres myTrainer_reproduction Task024_Promise 0 \
  -p nnUNetPlansv2.1_trgSp_z2.2_yx0.6125 --npz --max_num_epochs 200
```

> The results go to
> `$RESULTS_FOLDER/nnUNet/3d_fullres/Task024_Promise/myTrainer_reproduction__<plans_id>/fold_0/`

* To **resume**: add `--continue_training`.
* To **limit epochs** later: `--continue_training --max_num_epochs 50`.
* To avoid overwriting results: change **any** of `{trainer, plans_id, fold}`.

### 5.2 Loop over multiple spacings (fold 0)

```bash
zs=("1p0" "2p2" "4p0")
folds=(0)

for z in "${zs[@]}"; do
  plan="nnUNetData_plans_v2.1_trgSp_z${z}_yx0p6125"   # exact base name of your *_plans_3D.pkl
  for f in "${folds[@]}"; do
    echo "== Training $plan fold $f =="
    nnUNet_train 3d_fullres myTrainer_reproduction Task024_Promise "$f" \
      -p "$plan" --npz --max_num_epochs 200
  done
done
```

> If you stopped early and there’s no `model_final_checkpoint.model`, see §7.2.

---

## 6) Validation & metrics

### 6.1 Preferred: trainer’s validation (needs **final** checkpoint)

```bash
nnUNet_train 3d_fullres myTrainer_reproduction Task024_Promise 0 \
  -p nnUNetPlansv2.1_trgSp_z2.2_yx0.6125 --validation_only --npz
```

This will compute validation metrics and write them under the fold folder.

### 6.2 If you stopped early (no final checkpoint)

nnU-Net v1 expects `model_final_checkpoint.model`. You likely have `model_latest` or `model_best`. Two options:

**A) Copy latest → final**

```bash
FOLD_DIR="$RESULTS_FOLDER/nnUNet/3d_fullres/Task024_Promise/myTrainer_reproduction__nnUNetPlansv2.1_trgSp_z2.2_yx0.6125/fold_0"
cp "$FOLD_DIR/model_latest.model"     "$FOLD_DIR/model_final_checkpoint.model"
cp "$FOLD_DIR/model_latest.model.pkl" "$FOLD_DIR/model_final_checkpoint.model.pkl"

nnUNet_train 3d_fullres myTrainer_reproduction Task024_Promise 0 \
  -p nnUNetPlansv2.1_trgSp_z2.2_yx0.6125 --validation_only --npz
```

**B) Resume and stop immediately to write “final”**

```bash
nnUNet_train 3d_fullres myTrainer_reproduction Task024_Promise 0 \
  -p nnUNetPlansv2.1_trgSp_z2.2_yx0.6125 --continue_training --max_num_epochs <last_epoch_number>
# then run --validation_only
```

---

## 7) Prediction (inference)

### 7.1 On the test set

```bash
OUT="/teamspace/studios/this_studio/preds_z22_test"
nnUNet_predict \
  -i "$nnUNet_raw_data_base/nnUNet_raw_data/Task024_Promise/imagesTs" \
  -o "$OUT" \
  -t Task024_Promise -m 3d_fullres \
  -p nnUNetPlansv2.1_trgSp_z2.2_yx0.6125 \
  -tr myTrainer_reproduction \
  -f 0 -chk model_best
```

### 7.2 (Optional) Export to PROMISE submission format (.mhd)

The conversion helper in `Task024_Promise2012.py` can export `.nii.gz → .mhd`:

```python
# quick one-off snippet
from nnunet.dataset_conversion.Task024_Promise2012 import export_for_submission
export_for_submission("/teamspace/studios/this_studio/preds_z22_test", "/teamspace/studios/this_studio/preds_z22_mhd")
```

---

## 8) What to report (for the README/results section)

* **Plans ID** (encodes target spacing): e.g., `nnUNetPlansv2.1_trgSp_z2.2_yx0.6125`
* **Trainer**: `myTrainer_reproduction`
* **Fold(s)**: 0 (or 0–4 if you do 5-fold)
* **Epochs**, **batch size**, **patch size** (printed at start of training)
* **Validation Dice** (foreground/global foreground)
* **Throughput** (optional): time per epoch from logs

---

## 9) Troubleshooting

* **FileNotFoundError for `*_plans_3D.pkl`**: your `-p` must be the **basename** without `_plans_3D.pkl`. Confirm with:

  ```bash
  ls -1 "$nnUNet_preprocessed/Task024_Promise"/*_plans_3D.pkl
  ```
* **Overwrites**: training with the same `(network, trainer, task, plans_id, fold)` overwrites. Change `-p` or trainer name.
* **Windows path issues**: prefer WSL; otherwise keep paths short, avoid spaces/OneDrive.
* **“Final checkpoint not found”**: copy `model_latest` → `model_final_checkpoint` or resume and stop immediately (§6.2).

---

## 10) Example: full z-sweep end-to-end

```bash
# 0) env
conda activate mnet-repro
export nnUNet_raw_data_base=/teamspace/studios/this_studio/nnUNet_raw
export nnUNet_preprocessed=/teamspace/studios/this_studio/nnUNet_preprocessed
export RESULTS_FOLDER=/teamspace/studios/this_studio/nnUNet_results
export CUDA_VISIBLE_DEVICES=0

# 1) convert (one time)
python -m nnunet.dataset_conversion.Task024_Promise2012 \
  --source /teamspace/data/PROMISE12 --target "$nnUNet_raw_data_base/nnUNet_raw_data/Task024_Promise"

# 2) plan+preprocess for each spacing (assuming per-z planner classes exist)
nnUNet_plan_and_preprocess -t 24 -pl3d ExperimentPlanner3D_v21_trgSp_z1p0_yx0p6125 -tl 8 -tf 8
nnUNet_plan_and_preprocess -t 24 -pl3d ExperimentPlanner3D_v21_trgSp_z2p2_yx0p6125 -tl 8 -tf 8
nnUNet_plan_and_preprocess -t 24 -pl3d ExperimentPlanner3D_v21_trgSp_z4p0_yx0p6125 -tl 8 -tf 8

# 3) train (fold 0 for each)
for z in 1.0 2.2 4.0; do
  plan="nnUNetPlansv2.1_trgSp_z${z}_yx0.6125"
  nnUNet_train 3d_fullres myTrainer_reproduction Task024_Promise 0 -p "$plan" --npz --max_num_epochs 200
done

# 4) validate (uses final checkpoint; see §6.2 if you stopped early)
for z in 1.0 2.2 4.0; do
  plan="nnUNetPlansv2.1_trgSp_z${z}_yx0.6125"
  nnUNet_train 3d_fullres myTrainer_reproduction Task024_Promise 0 -p "$plan" --validation_only --npz
done

# 5) predict on test (z=2.2 example)
nnUNet_predict -i "$nnUNet_raw_data_base/nnUNet_raw_data/Task024_Promise/imagesTs" \
  -o "/teamspace/studios/this_studio/preds_z22_test" \
  -t Task024_Promise -m 3d_fullres \
  -p nnUNetPlansv2.1_trgSp_z2.2_yx0.6125 -tr myTrainer_reproduction -f 0 -chk model_best
```

