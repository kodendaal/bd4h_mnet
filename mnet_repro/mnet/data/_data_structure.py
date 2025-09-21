# Python
from pathlib import Path
import re
import csv, json, random
import SimpleITK as sitk

DATA_ROOT     = Path("./data")             # change if needed
PROMISE12_DIR = DATA_ROOT / "PROMISE12"
TRAIN_DIR     = PROMISE12_DIR / "training_data"
TEST_DIR      = PROMISE12_DIR / "test_data"

# Harmonized “raw” view we’ll generate (non-destructive)
RAW_ROOT      = PROMISE12_DIR / "raw"
RAW_IMG_TRAIN = RAW_ROOT / "images"
RAW_LBL_TRAIN = RAW_ROOT / "labels"
RAW_IMG_TEST  = RAW_ROOT / "images_test"
RAW_LBL_TEST  = RAW_ROOT / "labels_test"

for p in [PROMISE12_DIR, TRAIN_DIR, TEST_DIR, RAW_IMG_TRAIN, RAW_LBL_TRAIN, RAW_IMG_TEST, RAW_LBL_TEST]:
    p.mkdir(parents=True, exist_ok=True)

print("Using:")
print("  PROMISE12_DIR:", PROMISE12_DIR)
print("  TRAIN_DIR    :", TRAIN_DIR)
print("  TEST_DIR     :", TEST_DIR)
print("  RAW_ROOT     :", RAW_ROOT)


IMG_RE = re.compile(r"^(?P<case>Case\d+)\.mhd$", re.IGNORECASE)
SEG_RE = re.compile(r"^(?P<case>Case\d+)_segmentation\.mhd$", re.IGNORECASE)

def index_mhd(folder: Path):
    images, labels = {}, {}
    for p in folder.glob("*.mhd"):
        name = p.name
        m_img = IMG_RE.match(name)
        m_seg = SEG_RE.match(name)
        if m_img:
            case = m_img.group("case")
            images[case] = p
        elif m_seg:
            case = m_seg.group("case")
            labels[case] = p
    return images, labels

img_tr, lbl_tr = index_mhd(TRAIN_DIR)
img_te, lbl_te = index_mhd(TEST_DIR)

print(f"[training_data] images: {len(img_tr)} | labels: {len(lbl_tr)}")
print(f"[test_data]     images: {len(img_te)} | labels: {len(lbl_te)}  (may be 0 if GT not provided)")
print("Sample train cases:", sorted(list(img_tr.keys()))[:5])

def read_mhd_info(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # [D, H, W]
    spacing = img.GetSpacing()         # (sx, sy, sz)
    direction = img.GetDirection()     # len 9 for 3D
    origin = img.GetOrigin()
    return arr.shape, spacing, direction, origin

def report_pairs(imgs: dict, lbls: dict, split_name: str, out_dir: Path):
    rows = []
    for case, imgp in sorted(imgs.items()):
        shp_i, sp_i, dir_i, org_i = read_mhd_info(imgp)
        lblp = lbls.get(case, None)
        if lblp is not None:
            shp_l, sp_l, dir_l, org_l = read_mhd_info(lblp)
        else:
            shp_l = sp_l = dir_l = org_l = None

        rows.append({
            "split": split_name,
            "case": case,
            "image_path": str(imgp),
            "label_path": str(lblp) if lblp else "",
            "img_shape_DHW": f"{shp_i[0]}x{shp_i[1]}x{shp_i[2]}",
            "img_spacing_xyz": f"{sp_i[0]:.4f},{sp_i[1]:.4f},{sp_i[2]:.4f}",
            "has_label": int(lblp is not None),
            "lbl_shape_DHW": "" if shp_l is None else f"{shp_l[0]}x{shp_l[1]}x{shp_l[2]}",
            "lbl_spacing_xyz": "" if sp_l is None else f"{sp_l[0]:.4f},{sp_l[1]:.4f},{sp_l[2]:.4f}",
        })
    # write files
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"promise12_{split_name}_report.csv"
    json_path = out_dir / f"promise12_{split_name}_report.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["split","case"])
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))
    return csv_path, json_path, rows

REPORT_DIR = PROMISE12_DIR / "reports"
csv_tr, json_tr, rows_tr = report_pairs(img_tr, lbl_tr, "train", REPORT_DIR)
csv_te, json_te, rows_te = report_pairs(img_te, lbl_te, "test", REPORT_DIR)
print("Wrote:\n ", csv_tr, "\n ", csv_te)
print("Train cases:", len(rows_tr), " | Test cases:", len(rows_te))
if rows_tr[:3]:
    print("Sample train rows:", rows_tr[:3])


VAL_SPLIT_DIR = PROMISE12_DIR / "splits"
VAL_SPLIT_DIR.mkdir(exist_ok=True, parents=True)

def make_train_val_split(train_cases, seed=42, val_ratio=0.2):
    train_cases = sorted(train_cases)
    rng = random.Random(seed)
    rng.shuffle(train_cases)
    n_val = max(1, int(round(len(train_cases) * val_ratio)))
    val_cases = set(train_cases[:n_val])
    tr_cases = [c for c in train_cases if c not in val_cases]
    return tr_cases, sorted(list(val_cases))

train_cases = sorted(list(img_tr.keys()))
tr_cases, val_cases = make_train_val_split(train_cases, seed=42, val_ratio=0.2)

split_json = {
    "train": tr_cases,
    "val": val_cases,
    "test": sorted(list(img_te.keys()))
}
(SPLIT_FILE := VAL_SPLIT_DIR / "promise12_split_seed42_v20.json").write_text(json.dumps(split_json, indent=2))
print("Wrote split file:", SPLIT_FILE)
print("Counts -> train:", len(tr_cases), "val:", len(val_cases), "test:", len(split_json["test"]))


