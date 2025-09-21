import json, numpy as np
from pathlib import Path
import SimpleITK as sitk
from torch.utils.data import Dataset

from .patching import iter_sliding_windows, crop_patch, sample_random_patch
from .transforms_3d import rand_flip3d, rand_intensity_jitter, center_crop_or_pad_3d

def sitk_read_array(niftipath: Path):
    img = sitk.ReadImage(str(niftipath))
    arr = sitk.GetArrayFromImage(img)  # [D,H,W]
    return img, arr

class Promise12Patches(Dataset):
    def __init__(self, manifest_path, split, patch_DHW=(16,128,128), mode="random",
                 stride_DHW=(8,64,64), aug=None, max_patches_per_volume=None):
        self.manifest = json.loads(Path(manifest_path).read_text())
        assert split in ("train","val","test"), split
        self.entries = list(self.manifest[split])
        self.split = split
        self.patch_DHW = tuple(patch_DHW)
        self.mode = mode
        self.stride_DHW = tuple(stride_DHW)
        self.aug = aug or {}
        self.max_patches = max_patches_per_volume
        self._index = []
        if self.mode == "sliding":
            for i, e in enumerate(self.entries):
                _, img = sitk_read_array(Path(e["image"]))
                D,H,W = img.shape
                cnt=0
                # iterate true image coords; ensure coverage even if image < patch
                if D < self.patch_DHW[0] or H < self.patch_DHW[1] or W < self.patch_DHW[2]:
                    starts = [(0,0,0)]
                else:
                    starts = list(iter_sliding_windows((D,H,W), self.patch_DHW, self.stride_DHW))
                for zyx in starts:
                    self._index.append((i, zyx, (D,H,W)))
                    cnt += 1
                    if self.max_patches and cnt >= self.max_patches:
                        break
        else:
            self._index = list(range(len(self.entries)))

    def __len__(self):
        return len(self._index)

    def _load_case(self, entry):
        img_path = Path(entry["image"])
        lbl_path = Path(entry["label"]) if entry.get("label") else None
        _, img = sitk_read_array(img_path)
        lbl = None
        if lbl_path and lbl_path.exists():
            _, lbl = sitk_read_array(lbl_path)
            lbl = lbl.astype(np.int64)
        img = img.astype(np.float32)
        return img, lbl

    def _apply_aug(self, img, lbl):
        if self.split == "train":
            img = rand_intensity_jitter(img, p=self.aug.get("intensity_p", 0.2))
            img, lbl = rand_flip3d(img, lbl, p=self.aug.get("flip_p", 0.5))
        return img, lbl

    def __getitem__(self, idx):
        if self.mode == "sliding":
            entry_idx, start_zyx, img_shape = self._index[idx]
            entry = self.entries[entry_idx]
            img, lbl = self._load_case(entry)
            D,H,W = img.shape
            z0,y0,x0 = start_zyx
            pD,pH,pW = self.patch_DHW
            # crop valid region within image
            d_valid = min(pD, max(0, D - z0))
            h_valid = min(pH, max(0, H - y0))
            w_valid = min(pW, max(0, W - x0))
            img_crop = img[z0:z0+d_valid, y0:y0+h_valid, x0:x0+w_valid]
            lbl_crop = lbl[z0:z0+d_valid, y0:y0+h_valid, x0:x0+w_valid] if lbl is not None else None
            # pad to patch size (top-left-front)
            img_patch = np.zeros((pD,pH,pW), dtype=np.float32)
            img_patch[:d_valid, :h_valid, :w_valid] = img_crop
            if lbl_crop is not None:
                lbl_patch = np.zeros((pD,pH,pW), dtype=np.int64)
                lbl_patch[:d_valid, :h_valid, :w_valid] = lbl_crop
            else:
                lbl_patch = None
            # no aug for val/test
            sample = {
                "image": img_patch[np.newaxis,...].astype(np.float32),
                "label": lbl_patch,
                "case": entry["case"],
                "start": (int(z0),int(y0),int(x0)),
                "valid": (int(d_valid),int(h_valid),int(w_valid)),
                "img_shape": (int(D),int(H),int(W)),
                "label_path": entry.get("label", ""),     # NEW: so validator can read full GT
                "image_path": entry.get("image", "")      # optional, handy for debugging
}
            return sample
        else:
            entry = self.entries[idx]
            img, lbl = self._load_case(entry)
            D,H,W = img.shape
            pD,pH,pW = self.patch_DHW
            if D < pD or H < pH or W < pW:
                img, lbl = center_crop_or_pad_3d(img, self.patch_DHW, pad_value=0, lbl=lbl, lbl_pad_value=0)
                start = (0,0,0)
            else:
                start = sample_random_patch((D,H,W), self.patch_DHW)
            img_patch = crop_patch(img, start, self.patch_DHW)
            lbl_patch = crop_patch(lbl, start, self.patch_DHW) if lbl is not None else None
            img_patch, lbl_patch = self._apply_aug(img_patch, lbl_patch)
            return {
                "image": img_patch[np.newaxis,...].astype(np.float32),
                "label": (lbl_patch.astype(np.int64) if lbl_patch is not None else None),
                "case": entry["case"]
            }
