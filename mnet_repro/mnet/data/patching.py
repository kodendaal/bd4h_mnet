import numpy as np

def iter_sliding_windows(shape_DHW, patch_DHW, stride_DHW):
    D, H, W = shape_DHW
    pD, pH, pW = patch_DHW
    sD, sH, sW = stride_DHW
    d_starts = list(range(0, max(1, D - pD + 1), sD)) or [0]
    h_starts = list(range(0, max(1, H - pH + 1), sH)) or [0]
    w_starts = list(range(0, max(1, W - pW + 1), sW)) or [0]
    if d_starts[-1] != max(0, D - pD): d_starts.append(max(0, D - pD))
    if h_starts[-1] != max(0, H - pH): h_starts.append(max(0, H - pH))
    if w_starts[-1] != max(0, W - pW): w_starts.append(max(0, W - pW))
    for z in d_starts:
        for y in h_starts:
            for x in w_starts:
                yield (z, y, x)

def crop_patch(arr, start_zyx, patch_DHW):
    z,y,x = start_zyx
    pD,pH,pW = patch_DHW
    return arr[z:z+pD, y:y+pH, x:x+pW]

def sample_random_patch(shape_DHW, patch_DHW, rng=None):
    rng = rng or np.random
    D,H,W = shape_DHW
    pD,pH,pW = patch_DHW
    if D < pD: z0 = 0
    else: z0 = rng.randint(0, D - pD + 1)
    if H < pH: y0 = 0
    else: y0 = rng.randint(0, H - pH + 1)
    if W < pW: x0 = 0
    else: x0 = rng.randint(0, W - pW + 1)
    return (z0,y0,x0)
