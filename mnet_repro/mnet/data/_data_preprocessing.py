from pathlib import Path
import json, random, re
import textwrap
import numpy as np
import SimpleITK as sitk

DATA_ROOT     = Path("./data")                  # change if needed
PROMISE12_DIR = DATA_ROOT / "PROMISE12"
TRAIN_DIR     = PROMISE12_DIR / "training_data"
TEST_DIR      = PROMISE12_DIR / "test_data"

PREP_ROOT     = PROMISE12_DIR / "preprocessed"
(PREP_ROOT / "train" / "images").mkdir(parents=True, exist_ok=True)
(PREP_ROOT / "train" / "labels").mkdir(parents=True, exist_ok=True)
(PREP_ROOT / "val"   / "images").mkdir(parents=True, exist_ok=True)
(PREP_ROOT / "val"   / "labels").mkdir(parents=True, exist_ok=True)
(PREP_ROOT / "test"  / "images").mkdir(parents=True, exist_ok=True)
(PREP_ROOT / "test"  / "labels").mkdir(parents=True, exist_ok=True)  # may remain empty

SPLIT_DIR  = PROMISE12_DIR / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_FILE = SPLIT_DIR / "promise12_split_seed42_v20.json"   # use yours if you already created it

print("Using split file:", SPLIT_FILE)
print(SPLIT_FILE.read_text()[:500], "...\n")

# match both "Case00.mhd" and "Case00.MHD", and labels like "Case00_segmentation.mhd"
IMG_RE = re.compile(r"^(?P<case>Case\d+)\.(?:mhd|nii|nii\.gz)$", re.IGNORECASE)
SEG_RE = re.compile(r"^(?P<case>Case\d+)_segmentation\.(?:mhd|nii|nii\.gz)$", re.IGNORECASE)

def _find_pairs(folder):
    # defensive: accept str/Path; reject tuples/lists
    if isinstance(folder, (str, bytes)):
        folder = Path(folder)
    if not isinstance(folder, Path):
        raise TypeError(f"_find_pairs expected pathlib.Path or str, got {type(folder)}")

    images, labels = {}, {}
    # iterate all entries; rely on regex (case-insensitive)
    for p in folder.iterdir():
        if not p.is_file():
            continue
        m_img = IMG_RE.match(p.name)
        m_seg = SEG_RE.match(p.name)
        if m_img:
            case = m_img.group("case")
            images[case] = p
        elif m_seg:
            case = m_seg.group("case")
            labels[case] = p
    return images, labels

def read_sitk(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)   # [D,H,W]
    spacing = img.GetSpacing()          # (sx,sy,sz)
    direction = img.GetDirection()      # len=9
    origin = img.GetOrigin()
    return img, arr, spacing, direction, origin

def robust_intensity_norm(arr: np.ndarray, clip_percent=(1.0, 99.0), eps=1e-6):
    # \"\"\"Clip to robust range per-volume then z-score normalize (per volume).
    # Assumes arr is image intensities (float-ish). Returns float32.
    # \"\"\"
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, clip_percent)
    a = np.clip(a, lo, hi)
    mean = a.mean()
    std  = a.std() + eps
    a = (a - mean) / std
    return a

def write_nifti_like(src_img: sitk.Image, new_arr: np.ndarray, out_path: Path, is_label=False):
    # \"\"\"Write a NIfTI using src geometry but new pixel content.
    # For labels, cast to int16 (or uint8) and do NOT normalize.
    # \"\"\"
    new_img = sitk.GetImageFromArray(new_arr)
    new_img.SetSpacing(src_img.GetSpacing())
    new_img.SetDirection(src_img.GetDirection())
    new_img.SetOrigin(src_img.GetOrigin())
    if is_label:
        new_img = sitk.Cast(new_img, sitk.sitkInt16)
    else:
        new_img = sitk.Cast(new_img, sitk.sitkFloat32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(new_img, str(out_path))

def preprocess_case(image_mhd: Path, label_mhd: Path | None, out_img: Path, out_lbl: Path | None,
                    normalize=True):
    # Load image (arr: [D,H,W])
    src_img, img_arr, _, _, _ = read_sitk(image_mhd)

    # Normalize intensities (image only)
    if normalize:
        img_arr = robust_intensity_norm(img_arr)

    # Write image
    write_nifti_like(src_img, img_arr, out_img, is_label=False)

    # Write label if present (we assume label values {0,1} for PROMISE12)
    if label_mhd is not None and out_lbl is not None:
        _, lbl_arr, _, _, _ = read_sitk(label_mhd)
        lbl_arr = lbl_arr.astype(np.int16)
        write_nifti_like(src_img, lbl_arr, out_lbl, is_label=True)


def run_promise12_preprocess_verbose(train_dir, test_dir, split_file, out_root, normalize=True):
    imgs_tr, lbls_tr = _find_pairs(train_dir)
    imgs_te, lbls_te = _find_pairs(test_dir)
    split = json.loads(Path(split_file).read_text())
    manifest = {"train": [], "val": [], "test": []}

    def _emit(split_name, cases):
        for case in cases:
            img_src = (imgs_tr if split_name in ("train","val") else imgs_te).get(case)
            if img_src is None:
                print(f"[{split_name}] SKIP {case}: image not found at runtime")
                continue
            lbl_src = (lbls_tr if split_name in ("train","val") else lbls_te).get(case)
            out_img = Path(out_root) / split_name / "images" / f"{case}.nii.gz"
            out_lbl = (Path(out_root) / split_name / "labels" / f"{case}.nii.gz") if lbl_src is not None else None
            preprocess_case(img_src, lbl_src, out_img, out_lbl, normalize=normalize)
            print(f"[{split_name}] {case}: wrote image → {out_img.name} | label → {out_lbl.name if out_lbl else 'NONE'}")
            manifest[split_name].append({"case": case, "split": split_name, "image": str(out_img), "label": (str(out_lbl) if out_lbl else "")})

    (Path(out_root) / "train" / "images").mkdir(parents=True, exist_ok=True)
    (Path(out_root) / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (Path(out_root) / "val"   / "images").mkdir(parents=True, exist_ok=True)
    (Path(out_root) / "val"   / "labels").mkdir(parents=True, exist_ok=True)
    (Path(out_root) / "test"  / "images").mkdir(parents=True, exist_ok=True)
    (Path(out_root) / "test"  / "labels").mkdir(parents=True, exist_ok=True)

    _emit("train", split.get("train", []))
    _emit("val",   split.get("val", []))
    _emit("test",  split.get("test", []))

    (Path(out_root) / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest

train_dir  =Path("./data/PROMISE12/training_data")
test_dir   =Path("./data/PROMISE12/test_data")
split_file =Path("./data/PROMISE12/splits/promise12_split_seed42_v20.json")
out_root   =Path("./data/PROMISE12/preprocessed")
manifest = run_promise12_preprocess_verbose(
    train_dir=train_dir,
    test_dir=test_dir,
    split_file=split_file,
    out_root=out_root,
    normalize=True
)

print("Summary:", len(manifest["train"]), len(manifest["val"]), len(manifest["test"]))
