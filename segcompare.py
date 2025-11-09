#!/usr/bin/env python3
import argparse, os, json, glob, pathlib, csv, warnings, re
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from PIL import Image

try:
    import nibabel as nib
except ImportError as e:
    raise SystemExit("Please install nibabel: pip install nibabel") from e

# Optional for 3D export
_HAS_SKIMAGE = True
try:
    from skimage import measure
except Exception:
    _HAS_SKIMAGE = False
try:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except Exception:
    # If this import failed, matplotlib installation is busted; keep running without 3D renders
    Poly3DCollection = None

# ---------- utils ----------
def _ensure_dir(p: str): os.makedirs(p, exist_ok=True)

IMG_EXTS = (".png",".tif",".tiff",".jpg",".jpeg",".npy",".nii",".nii.gz")

def _is_img(p: str) -> bool:
    pl = p.lower(); return pl.endswith(IMG_EXTS)

def _id_from(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".nii.gz"): return name[:-7]
    return os.path.splitext(name)[0]

def _load_mask(path: str) -> np.ndarray:
    pl = path.lower()
    if pl.endswith(".npy"):
        arr = np.load(path)
        if arr.ndim == 2: return arr.astype(np.int64)
        if arr.ndim == 3:
            if arr.shape[0] < 8 and (arr.shape[0] != arr.shape[1] or arr.shape[0] != arr.shape[2]):
                arr = np.moveaxis(arr, 0, -1)
            return np.argmax(arr, axis=-1).astype(np.int64)
        if arr.ndim == 4:
            return np.argmax(arr, axis=-1).astype(np.int64)
        raise ValueError(f"Unsupported npy shape {arr.shape} for {path}")
    if pl.endswith(".nii") or pl.endswith(".nii.gz"):
        img = nib.load(path)
        data = img.get_fdata(dtype=np.float32)
        if data.ndim == 3:
            if np.all(np.isfinite(data)):
                if np.all(np.abs(data - np.rint(data)) < 1e-3):
                    data = np.rint(data)
            return data.astype(np.int64)
        if data.ndim == 4:
            return np.argmax(data, axis=-1).astype(np.int64)
        raise ValueError(f"Unsupported NIfTI shape {data.shape} for {path}")
    im = Image.open(path)
    return np.array(im)

def _load_image_float(path: str) -> np.ndarray:
    """
    Load a raw image volume as float32 for visualization.
    Supports NIfTI 3D or 4D (takes channel 0 if 4D).
    Returns ndarray with shape (X,Y,Z).
    """
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 3:
        return data
    if data.ndim == 4:
        # assume channels last (X,Y,Z,C) -> take C0
        if data.shape[-1] <= 8:
            return data[..., 0]
        # channels first (C,X,Y,Z)
        if data.shape[0] <= 8:
            return data[0]
    raise ValueError(f"Unsupported raw image shape {data.shape} for {path}")

def _read_set(path: Optional[str]) -> Optional[set]:
    if not path or not os.path.isfile(path): return None
    with open(path, "r") as f:
        return {ln.strip() for ln in f if ln.strip()}

# ---------- discovery ----------
def _scan_files(glob_pattern: str) -> Dict[str,str]:
    files = {}; paths = []
    for pat in _expand_braces(glob_pattern):
        paths.extend(glob.glob(pat, recursive=True))
    for p in paths:
        if _is_img(p):
            files[_id_from(p)] = p
    return files

def _expand_braces(pattern: str) -> List[str]:
    if "{" not in pattern: return [pattern]
    pre = pattern[:pattern.index("{")]
    rest = pattern[pattern.index("{")+1:]
    opts = rest[:rest.index("}")].split(",")
    post = rest[rest.index("}")+1:]
    out = []
    for o in opts:
        out.extend(_expand_braces(pre + o + post))
    return out

def _scan_model_preds(model_dir: str) -> Dict[str,str]:
    mapping = {}
    for p in glob.glob(os.path.join(model_dir, "**", "*"), recursive=True):
        if _is_img(p): mapping[_id_from(p)] = p
    return mapping

def _read_manifest(path: str) -> Tuple[Dict[str,str], Dict[str,Dict[str,str]]]:
    ids_to_gt = {}; ids_to_model = {}
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        fields = rdr.fieldnames or []
        model_cols = [c for c in fields if c not in ("id","gt")]
        for row in rdr:
            i = row["id"].strip()
            ids_to_gt[i] = row["gt"].strip()
            ids_to_model[i] = {m: row[m].strip() for m in model_cols if row.get(m,"").strip()}
    return ids_to_gt, ids_to_model

# ---------- raw image location ----------
TASK_DIR_RE = re.compile(r"^Task\d+_.+$")

def _find_task_dir_from_gt(gt_path: str) -> Optional[str]:
    """
    Walk up from a GT path and return the TaskXXX_* directory name if found.
    """
    p = pathlib.Path(gt_path).resolve()
    for parent in [p] + list(p.parents):
        name = parent.name
        if TASK_DIR_RE.match(name):
            return name
    return None

def _try_load_raw_slice(case_id: str,
                        gt_path: str,
                        raw_roots: Optional[List[str]]) -> Optional[np.ndarray]:
    """
    Attempt to load the raw image center slice for a case from nnUNet_raw_data/<Task>/imagesTr/<id>_0000.nii.gz
    Returns 2D array normalized to [0,1], or None if not found or on failure.
    """
    task_dir = _find_task_dir_from_gt(gt_path)  # e.g., Task024_Promise or Task029_LITS
    candidates = []

    names_to_try = []
    if task_dir:
        names_to_try.append(task_dir)

    candidate_roots = list(raw_roots or [])
    if not candidate_roots:
        candidate_roots = [
            "/teamspace/studios/this_studio/nnUNet_raw/nnUNet_raw_data",
            "/teamspace/studios/this_studio/nnUNet_raw_data",
        ]

    for root in candidate_roots:
        for tname in names_to_try or [""]:
            if not tname: continue
            fpath = os.path.join(root, tname, "imagesTr", f"{case_id}_0000.nii.gz")
            candidates.append(fpath)

    for fp in candidates:
        if os.path.exists(fp):
            try:
                vol = _load_image_float(fp)  # (X,Y,Z)
                z = vol.shape[-1] // 2
                sl = vol[..., z].astype(np.float32)
                vmin, vmax = np.percentile(sl, 1), np.percentile(sl, 99)
                if vmax > vmin:
                    sl = np.clip((sl - vmin) / (vmax - vmin), 0, 1)
                else:
                    sl = sl - sl.min()
                    denom = sl.max() - sl.min()
                    sl = sl / (denom + 1e-8)
                return sl
            except Exception as e:
                warnings.warn(f"Failed to load raw image {fp}: {e}")
                return None
    return None

# ---------- metrics (Dice-only) ----------
def _confusion(gt: np.ndarray, pr: np.ndarray, num_classes: int) -> np.ndarray:
    if gt.shape != pr.shape:
        raise ValueError(f"Shape mismatch GT {gt.shape} vs Pred {pr.shape}")
    mask = (gt >= 0) & (gt < num_classes)
    x = num_classes * gt[mask].astype(np.int64) + pr[mask].astype(np.int64)
    return np.bincount(x, minlength=num_classes**2).reshape(num_classes, num_classes)

def _perclass_dice(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    pos_gt = cm.sum(axis=1).astype(np.float64)
    pos_pred = cm.sum(axis=0).astype(np.float64)
    denom = pos_gt + pos_pred
    with np.errstate(divide='ignore', invalid='ignore'):
        dice = np.where(denom > 0, 2.0 * tp / denom, np.nan)
    return dice

def _metrics(cm: np.ndarray, foreground_labels: List[int]) -> Dict[str,object]:
    dice = _perclass_dice(cm)
    fg_vals = dice[foreground_labels] if len(foreground_labels) else np.array([])
    mean_dice = float(np.nanmean(fg_vals)) if fg_vals.size else float("nan")
    return {
        "meanDice": mean_dice,
        "_perClassDice": dice.tolist(),
    }

# ---------- visualization helpers (Raw | GT | Pred | Errors with legend) ----------
def _make_lut(num_classes: int) -> np.ndarray:
    lut = np.zeros((num_classes, 3), dtype=np.float32)
    lut[0] = np.array([0.7, 0.7, 0.7], np.float32)  # background
    if num_classes > 1:
        tab = plt.cm.get_cmap("tab20", 20)
        for c in range(1, num_classes):
            lut[c] = tab((c - 1) % 20)[:3]
    return lut

def _colorize(mask2d: np.ndarray, lut: np.ndarray) -> np.ndarray:
    mask2d = np.asarray(mask2d, dtype=np.int64)
    mask2d = np.clip(mask2d, 0, lut.shape[0]-1)
    return lut[mask2d]

def _error_map(gt2d: np.ndarray, pr2d: np.ndarray) -> np.ndarray:
    gt_fg = gt2d > 0
    pr_fg = pr2d > 0
    tp = gt_fg & pr_fg & (gt2d == pr2d)
    fp = (~gt_fg) & pr_fg
    fn = gt_fg & (~pr_fg)
    mis = gt_fg & pr_fg & (gt2d != pr2d)
    tn = (~gt_fg) & (~pr_fg)

    h, w = gt2d.shape
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[tn]  = [0.15, 0.15, 0.15]
    rgb[tp]  = [0.0, 1.0, 1.0]
    rgb[fp]  = [1.0, 0.0, 0.0]
    rgb[fn]  = [1.0, 0.5, 0.0]
    rgb[mis] = [1.0, 1.0, 0.0]
    return rgb

def _compose_preview(gt2d: np.ndarray,
                     pr2d: np.ndarray,
                     num_classes: int,
                     save_path: str,
                     raw2d: Optional[np.ndarray] = None):
    lut = _make_lut(num_classes)
    gt_rgb = _colorize(gt2d, lut)
    pr_rgb = _colorize(pr2d, lut)
    err_rgb = _error_map(gt2d, pr2d)

    panels = []
    titles = []
    if raw2d is not None:
        g = raw2d.astype(np.float32)
        g = np.clip(g, 0, 1)
        raw_rgb = np.stack([g, g, g], axis=-1)
        panels.append(raw_rgb)
        titles.append("Raw")

    panels.extend([gt_rgb, pr_rgb, err_rgb])
    titles.extend(["Ground Truth", "Prediction", "Errors"])

    ncols = len(panels)
    fig, axs = plt.subplots(1, ncols, figsize=(4*ncols, 4))
    if ncols == 1:
        axs = [axs]

    for ax, panel, title in zip(axs, panels, titles):
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")

    # Build legend
    lut_full = _make_lut(num_classes)
    patches = [mpatches.Patch(color=lut_full[0], label="background")]
    max_show = min(num_classes-1, 12)
    for c in range(1, 1 + max_show):
        patches.append(mpatches.Patch(color=lut_full[c], label=f"class {c}"))
    patches.extend([
        mpatches.Patch(color=[0.0,1.0,1.0], label="TP (match>0)"),
        mpatches.Patch(color=[1.0,0.0,0.0], label="FP (pred only)"),
        mpatches.Patch(color=[1.0,0.5,0.0], label="FN (gt only)"),
        mpatches.Patch(color=[1.0,1.0,0.0], label="Mislabel (gt≠pred>0)"),
    ])
    fig.legend(handles=patches, loc="lower center", ncol=min(6, len(patches)), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _center_axial(sliceable3d: np.ndarray) -> np.ndarray:
    z = sliceable3d.shape[-1]//2
    return sliceable3d[..., z]

def _save(arr, path):
    if arr.dtype!=np.uint8: arr=(arr*255).clip(0,255).astype(np.uint8)
    Image.fromarray(arr).save(path)

# ---------- 3D export helpers (optional) ----------
def _nifti_spacing(path: str) -> Optional[Tuple[float,float,float]]:
    try:
        if path.lower().endswith((".nii", ".nii.gz")):
            img = nib.load(path)
            z = img.header.get_zooms()
            if len(z) >= 3:
                return float(z[0]), float(z[1]), float(z[2])
    except Exception:
        pass
    return None

def _marching_cubes_binary(vol: np.ndarray, spacing: Optional[Tuple[float,float,float]]):
    if not _HAS_SKIMAGE:
        raise RuntimeError("scikit-image is required for 3D mesh export. Install: pip install scikit-image")
    v = vol.astype(np.uint8)
    v = np.transpose(v, (2, 0, 1))  # (H,W,D) -> (Z,Y,X)
    sp = spacing if spacing is not None else (1.0, 1.0, 1.0)
    verts, faces, _, _ = measure.marching_cubes(v, level=0.5, spacing=sp)
    return verts, faces

def _save_stl(verts: np.ndarray, faces: np.ndarray, out_path: str):
    import struct
    with open(out_path, "wb") as f:
        f.write(b"segcompare STL export".ljust(80, b"\0"))
        f.write(struct.pack("<I", faces.shape[0]))
        v0 = verts[faces[:,0]]
        v1 = verts[faces[:,1]]
        v2 = verts[faces[:,2]]
        n = np.cross(v1 - v0, v2 - v0)
        n_norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        n = n / n_norm
        for k in range(faces.shape[0]):
            f.write(struct.pack("<3f", *n[k]))
            f.write(struct.pack("<3f", *v0[k]))
            f.write(struct.pack("<3f", *v1[k]))
            f.write(struct.pack("<3f", *v2[k]))
            f.write(struct.pack("<H", 0))

def _render_mesh_quick(meshes: List[Tuple[np.ndarray, np.ndarray]], save_path: str, elev=20, azim=45):
    if Poly3DCollection is None:
        warnings.warn("Matplotlib 3D backend unavailable; skipping mesh snapshot.")
        return
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection="3d")
    for vi, fi in meshes:
        coll = Poly3DCollection(vi[fi], linewidths=0.05, alpha=0.8)
        ax.add_collection3d(coll)
    if meshes:
        all_verts = np.vstack([m[0] for m in meshes])
    else:
        all_verts = np.zeros((1,3))
    mins = all_verts.min(axis=0); maxs = all_verts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() * 0.6 + 1e-6
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def _export_3d_for_case(case_id: str, pr: np.ndarray, num_classes: int, pred_path: str, out_dir: str):
    spacing = _nifti_spacing(pred_path)
    case_dir = os.path.join(out_dir, "meshes", case_id)
    _ensure_dir(case_dir)

    meshes_for_render = []
    for cls in range(1, num_classes):
        mask = (pr == cls)
        if not np.any(mask):
            continue
        try:
            verts, faces = _marching_cubes_binary(mask, spacing)
            stl_path = os.path.join(case_dir, f"class_{cls}.stl")
            _save_stl(verts, faces, stl_path)
            meshes_for_render.append((verts, faces))
        except Exception as e:
            warnings.warn(f"Marching cubes failed for {case_id}, class {cls}: {e}")

    if meshes_for_render:
        png_path = os.path.join(case_dir, f"{case_id}_3d.png")
        _render_mesh_quick(meshes_for_render, png_path)

# ---------- runner ----------
def evaluate(
    out_dir: str,
    num_classes: int,
    model_dirs: List[str],
    segments_dir: Optional[str]=None,
    gt_dir: Optional[str]=None,
    gt_glob: Optional[str]=None,
    pred_glob: Optional[str]=None,
    manifest: Optional[str]=None,
    limit: Optional[int]=None,
    raw_roots: Optional[List[str]] = None,
    save_3d: bool = False,
):
    _ensure_dir(out_dir)

    # 1) build id -> gt path
    if manifest:
        ids_to_gt, ids_to_model = _read_manifest(manifest)
        ids = list(ids_to_gt.keys())
    else:
        if gt_glob:
            gt_map = _scan_files(gt_glob)
        else:
            if not gt_dir: raise ValueError("Provide --gt_dir or --gt_glob or --manifest")
            gt_map = {}
            for p in glob.glob(os.path.join(gt_dir, "**", "*"), recursive=True):
                if _is_img(p): gt_map[_id_from(p)] = p
        ids = sorted(gt_map.keys())

    if limit: ids = ids[:limit]

    # 2) segments
    segments = {"all": None}
    if segments_dir and os.path.isdir(segments_dir):
        for p in glob.glob(os.path.join(segments_dir, "*.txt")):
            segments[pathlib.Path(p).stem] = _read_set(p)

    # 3) model prediction maps
    model_predmaps: Dict[str, Dict[str,str]] = {}
    for mdir in model_dirs:
        mname = pathlib.Path(mdir).name
        if manifest:
            predmap = {i: ids_to_model.get(i,{}).get(mname,"") for i in ids}
        else:
            if pred_glob:
                pat = pred_glob.format(model=mdir, model_name=mname)
                predmap = _scan_files(pat)
            else:
                predmap = _scan_model_preds(mdir)
        model_predmaps[mname] = predmap

    # 4) compute
    results = {}
    all_model_summaries = {}
    foreground_labels = list(range(1, num_classes))

    for mname, predmap in model_predmaps.items():
        model_out = os.path.join(out_dir, mname); _ensure_dir(model_out)
        _ensure_dir(os.path.join(model_out, "previews"))
        seg_cm = {s: np.zeros((num_classes,num_classes), np.int64) for s in segments}
        per_img = []

        for i in ids:
            gt_path = (ids_to_gt[i] if manifest else gt_map.get(i))
            pr_path = predmap.get(i)
            if not gt_path or not os.path.exists(gt_path) or not pr_path or not os.path.exists(pr_path):
                continue

            gt = _load_mask(gt_path)
            pr = _load_mask(pr_path)

            if gt.shape != pr.shape:
                raise ValueError(f"[{mname}] shape mismatch for {i}: GT {gt.shape} vs Pred {pr.shape}")

            cm = _confusion(gt, pr, num_classes)
            for s, sset in segments.items():
                if sset is None or i in sset:
                    seg_cm[s] += cm

            # metrics (Dice only)
            m = _metrics(cm, foreground_labels)

            # previews (Raw | GT | Pred | Errors)
            try:
                gt2d = _center_axial(gt)
                pr2d = _center_axial(pr)
                raw2d = _try_load_raw_slice(i, gt_path, raw_roots)
                prev_path = os.path.join(model_out, "previews", f"{i}_preview.png")
                _compose_preview(gt2d, pr2d, num_classes, prev_path, raw2d=raw2d)
            except Exception as e:
                warnings.warn(f"Preview failed for {i}: {e}")

            # optional 3D export
            if save_3d:
                try:
                    _export_3d_for_case(i, pr, num_classes, pr_path, model_out)
                except Exception as e:
                    warnings.warn(f"3D export failed for {i}: {e}")

            per_img.append({
                "id": i,
                "meanDice": m["meanDice"],
                "_perClassDice": m["_perClassDice"],
            })

        # aggregate + Dice plots
        seg_metrics = {s: _metrics(cmacc, foreground_labels) for s,cmacc in seg_cm.items()}
        results[mname] = seg_metrics
        all_model_summaries[mname] = per_img

        labels = list(seg_metrics.keys())
        vals = [seg_metrics[k]["meanDice"] for k in labels]
        plt.figure(figsize=(6,4))
        plt.bar(range(len(labels)), vals)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("meanDice (foreground)")
        plt.title(f"{mname} — meanDice by segment")
        plt.tight_layout()
        plt.savefig(os.path.join(model_out, f"{mname}_meanDice_by_segment.png"), dpi=150)
        plt.close()

        ref = "all" if "all" in seg_metrics else labels[0]
        dices = seg_metrics[ref]["_perClassDice"]
        plt.figure(figsize=(6,4))
        plt.bar(range(len(dices)), dices)
        plt.xticks(range(len(dices)), [f"c{i}" for i in range(len(dices))])
        plt.ylabel("Dice")
        plt.title(f"{mname} — per-class Dice ({ref})")
        plt.tight_layout()
        plt.savefig(os.path.join(model_out, f"{mname}_per_class_dice.png"), dpi=150)
        plt.close()

        with open(os.path.join(model_out, "per_image.json"), "w") as f:
            json.dump(per_img, f, indent=2)

    # cross-model comparison (meanDice on 'all')
    names = list(results.keys())
    vals = [results[n].get("all", next(iter(results[n].values())))["meanDice"] for n in names]
    plt.figure(figsize=(7,4))
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.ylabel("meanDice (foreground)")
    plt.title("Model comparison — meanDice (segment: all)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "models_meanDice_all.png"), dpi=150)
    plt.close()

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--segments_dir", default=None)
    ap.add_argument("--gt_dir", default=None)
    ap.add_argument("--gt_glob", default=None, help="e.g. 'data/gt_niftis/**/*.nii.gz'")
    ap.add_argument("--pred_glob", default=None, help="e.g. '{model}/**/validation_raw_postprocessed/**/*.nii.gz'")
    ap.add_argument("--manifest", default=None, help="CSV with columns: id,gt,<modelname>...")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--raw_roots", nargs="*", default=None,
                    help="Optional search roots for nnUNet_raw_data (will look for <root>/<Task>/imagesTr/<id>_0000.nii.gz).")
    ap.add_argument("--save_3d", action="store_true",
                    help="Export per-class STL meshes and a quick 3D PNG per case")
    args = ap.parse_args()

    evaluate(
        out_dir=args.out_dir,
        num_classes=args.num_classes,
        model_dirs=args.models,
        segments_dir=args.segments_dir,
        gt_dir=args.gt_dir,
        gt_glob=args.gt_glob,
        pred_glob=args.pred_glob,
        manifest=args.manifest,
        limit=args.limit,
        raw_roots=args.raw_roots,
        save_3d=args.save_3d,
    )

if __name__ == "__main__":
    main()
