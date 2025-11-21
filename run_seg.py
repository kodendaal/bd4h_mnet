# from segcompare import evaluate


# # ROOT = "/teamspace/studios/this_studio/nnUNet_results/nnUNet/3d_fullres/Task024_Promise/myTrainer_repoduction__nnUNetPlansv2.1_trgSp_z2p2_yx0p6125_150e_OW"
# # ROOT = "/teamspace/studios/this_studio/nnUNet_results/nnUNet/3d_fullres/Task024_Promise/myTrainer_reproduction__nnUNetData_plans_v2.1_trgSp_z2p2_yx0p6125_150e_OW"

# # Lits
# ROOT = "/teamspace/studios/this_studio/nnUNet_results/nnUNet/3d_fullres/Task029_LITS/myTrainer_reproduction__nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_150e_OW"

# gt_glob   = f"{ROOT}/gt_niftis/*.nii.gz"
# pred_glob = f"{ROOT}/fold_0/validation_raw_postprocessed/*.nii.gz"

# evaluate(
#     out_dir=f"{ROOT}/viz_results",
#     num_classes=3,                 # 0=background, 1=foreground
#     model_dirs=[f"{ROOT}/fold_0"], # just used for labeling; safe to include
#     gt_glob=gt_glob,
#     pred_glob=pred_glob,           # no {model} placeholder
# )


#!/usr/bin/env python3
from segcompare import evaluate

# --------- CONFIG (LiTS) ---------

# ROOT = "/teamspace/studios/this_studio/nnUNet_results/nnUNet/3d_fullres/Task024_Promise/myTrainer_reproduction__nnUNetData_plans_v2.1_trgSp_z2p2_yx0p6125_150e_OW"


# ROOT points at the LiTS task results directory
ROOT = "/teamspace/studios/this_studio/nnUNet_results/nnUNet/3d_fullres/Task029_LITS/myTrainer_reproduction__nnUNetData_plans_v2.1_trgSp_z1p0_yx0p9121_150e_I1_SP"

# Ground-truth NIfTIs (created earlier in your structure)
gt_glob   = f"{ROOT}/gt_niftis/*.nii.gz"

# Model predictions for the validation set
pred_glob = f"{ROOT}/fold_0/validation_raw_postprocessed/*.nii.gz"

if __name__ == "__main__":
    evaluate(
        out_dir=f"{ROOT}/viz_results",
        num_classes=3,                  # classes: 0=background, 1=liver, 2=tumor
        model_dirs=[f"{ROOT}/fold_0"],  # label for plots/outputs
        gt_glob=gt_glob,
        pred_glob=pred_glob,            # direct glob (no {model} placeholder needed)
        save_3d=False
    )
