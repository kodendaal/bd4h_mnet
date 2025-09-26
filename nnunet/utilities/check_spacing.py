import pickle, os
import numpy as np

p = os.path.expandvars("/teamspace/studios/this_studio/nnUNet_preprocessed/Task024_Promise/nnUNetData_plans_v2.1_trgSp_z1p0_yx0p6125_plans_3D.pkl")
plans = pickle.load(open(p, "rb"))

stage0 = plans["plans_per_stage"][0]
print("target/current spacing (z,y,x):", stage0["current_spacing"])
print("patch_size (z,y,x):", stage0["patch_size"])
print("batch_size:", stage0["batch_size"])
print("do_dummy_2D_data_aug:", stage0["do_dummy_2D_data_aug"])
print("observed median spacing (dataset):", np.mean(plans["dataset_properties"]["all_spacings"],axis=0), "... etc")