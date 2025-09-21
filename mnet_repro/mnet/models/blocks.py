import torch.nn as nn

def conv_norm_act(in_ch, out_ch, k=3, s=1, p=1, dim=3):
    if dim == 3:
        conv = nn.Conv3d(in_ch, out_ch, k, s, p, bias=False)
        norm = nn.InstanceNorm3d(out_ch, affine=True)
    elif dim == 2:
        conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        norm = nn.InstanceNorm2d(out_ch, affine=True)
    else:
        raise ValueError("dim must be 2 or 3")
    return nn.Sequential(conv, norm, nn.ReLU(inplace=True))
