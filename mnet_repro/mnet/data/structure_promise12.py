import re, csv, json, random
from pathlib import Path
import SimpleITK as sitk

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

def read_mhd_info(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)   # [D,H,W]
    spacing = img.GetSpacing()          # (sx,sy,sz)
    direction = img.GetDirection()      # len 9
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
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"promise12_{split_name}_report.csv"
    json_path = out_dir / f"promise12_{split_name}_report.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["split","case"])
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))
    return csv_path, json_path, rows

def make_train_val_split(train_cases, seed=42, val_ratio=0.2):
    train_cases = sorted(train_cases)
    rng = random.Random(seed)
    rng.shuffle(train_cases)
    n_val = max(1, int(round(len(train_cases) * val_ratio)))
    val_cases = set(train_cases[:n_val])
    tr_cases = [c for c in train_cases if c not in val_cases]
    return tr_cases, sorted(list(val_cases))

def write_split_file(train_cases, test_cases, split_dir: Path, seed=42, val_ratio=0.2, name="promise12_split_seed42_v20.json"):
    split_dir.mkdir(parents=True, exist_ok=True)
    tr, val = make_train_val_split(train_cases, seed=seed, val_ratio=val_ratio)
    split = {"train": tr, "val": val, "test": sorted(list(test_cases))}
    out = split_dir / name
    out.write_text(json.dumps(split, indent=2))
    return out

if __name__ == "__main__":
    # Minimal CLI-like usage:
    DATA_ROOT     = Path("./data")
    PROMISE12_DIR = DATA_ROOT / "PROMISE12"
    TRAIN_DIR     = PROMISE12_DIR / "training_data"
    TEST_DIR      = PROMISE12_DIR / "test_data"

    REPORT_DIR = PROMISE12_DIR / "reports"
    SPLIT_DIR  = PROMISE12_DIR / "splits"

    imgs_tr, lbls_tr = index_mhd(TRAIN_DIR)
    imgs_te, lbls_te = index_mhd(TEST_DIR)

    csv_tr, json_tr, rows_tr = report_pairs(imgs_tr, lbls_tr, "train", REPORT_DIR)
    csv_te, json_te, rows_te = report_pairs(imgs_te, lbls_te, "test",  REPORT_DIR)

    split_path = write_split_file(list(imgs_tr.keys()), list(imgs_te.keys()), split_dir=SPLIT_DIR, seed=42, val_ratio=0.2)
    print("Reports:", csv_tr, csv_te)
    print("Split file:", split_path)
