import numpy as np

def rand_flip3d(arr_img, arr_lbl=None, p=0.5, axes=(0,1,2), rng=None):
    """Random flips along given axes. axes are indices over (D,H,W)."""
    rng = rng or np.random
    img, lbl = arr_img, arr_lbl
    for ax in axes:
        if rng.rand() < p:
            img = np.flip(img, axis=ax).copy()
            if lbl is not None:
                lbl = np.flip(lbl, axis=ax).copy()
    return img, lbl

def rand_intensity_jitter(img, p=0.2, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1), rng=None):
    """Mild per-volume intensity jitter on normalized image."""
    rng = rng or np.random
    out = img
    if rng.rand() < p:
        s = rng.uniform(*scale_range)
        t = rng.uniform(*shift_range)
        out = out * s + t
    return out

def center_crop_or_pad_3d(img, target, pad_value=0, lbl=None, lbl_pad_value=0):
    """Crop or pad to exact target shape (D,H,W)."""
    D, H, W = img.shape
    tD, tH, tW = target
    out_img = np.full((tD,tH,tW), pad_value, dtype=img.dtype)
    if lbl is not None:
        out_lbl = np.full((tD,tH,tW), lbl_pad_value, dtype=lbl.dtype)
    # compute src/dst slices
    d0 = max(0, (D - tD)//2); d1 = d0 + min(D, tD)
    h0 = max(0, (H - tH)//2); h1 = h0 + min(H, tH)
    w0 = max(0, (W - tW)//2); w1 = w0 + min(W, tW)

    td0 = max(0, (tD - D)//2); td1 = td0 + (d1 - d0)
    th0 = max(0, (tH - H)//2); th1 = th0 + (h1 - h0)
    tw0 = max(0, (tW - W)//2); tw1 = tw0 + (w1 - w0)

    out_img[td0:td1, th0:th1, tw0:tw1] = img[d0:d1, h0:h1, w0:w1]
    if lbl is None:
        return out_img, None
    out_lbl[td0:td1, th0:th1, tw0:tw1] = lbl[d0:d1, h0:h1, w0:w1]
    return out_img, out_lbl
