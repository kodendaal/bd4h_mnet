import torch

class VolumeLogitAggregator:
    def __init__(self, vol_shape_DHW, num_classes, device="cpu", dtype=torch.float32):
        D,H,W = vol_shape_DHW
        self.sum_logits = torch.zeros((num_classes, D, H, W), dtype=dtype, device=device)
        self.counts     = torch.zeros((1, D, H, W), dtype=dtype, device=device)

    def add(self, patch_logits: torch.Tensor, start_zyx, valid_dhw):
        """
        patch_logits: [C, d, h, w] (full patch-size, padded)
        start_zyx: (z0,y0,x0) -- placement in the ORIGINAL image canvas
        valid_dhw: (d_valid,h_valid,w_valid) -- how much of patch is actual data
        """
        C, d, h, w = patch_logits.shape
        z0,y0,x0 = start_zyx
        dv,hv,wv = valid_dhw
        if dv<=0 or hv<=0 or wv<=0:
            return
        self.sum_logits[:, z0:z0+dv, y0:y0+hv, x0:x0+wv] += patch_logits[:, :dv, :hv, :wv]
        self.counts[:,     z0:z0+dv, y0:y0+hv, x0:x0+wv] += 1.0

    def get_avg(self):
        counts = torch.clamp(self.counts, min=1.0)
        return self.sum_logits / counts
