# Clean-room reimplementation that mirrors the official MNet's module graph and behaviors.
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Core blocks (InstanceNorm + LeakyReLU), anisotropy-aware ----------
def _act(leak=0.1): return nn.LeakyReLU(negative_slope=leak, inplace=True)

def _norm3d(c): return nn.InstanceNorm3d(c, affine=True)
def _norm2d(c): return nn.InstanceNorm2d(c, affine=True)

def _conv3d(in_c, out_c, k, p, s=1, bias=False):
    # k can be int (3 -> (3,3,3)) or tuple (1,3,3)
    if isinstance(k, int): k = (k, k, k)
    return nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)

class CB3d(nn.Module):
    """
    Two Conv-Norm-LReLU layers.
    kSize can be:
      - (3,3) meaning 3x3x3 twice (official shorthand)
      - ((1,3,3),(1,3,3)) meaning two anisotropic convs
    padding: either int (e.g., 1) or tuple matching k.
    """
    def __init__(self, in_channels, out_channels, kSize, stride=(1,1), padding=(1,1,1),
                 norm_args=(None, None), activation_args=(None, None)):
        super().__init__()
        if isinstance(kSize, tuple) and len(kSize) == 2 and isinstance(kSize[0], tuple):
            k1, k2 = kSize
            p1 = (0,1,1) if k1 == (1,3,3) else tuple((kk-1)//2 for kk in k1)
            p2 = (0,1,1) if k2 == (1,3,3) else tuple((kk-1)//2 for kk in k2)
        else:
            # e.g., (3,3) -> interpret as two 3x3x3 convs
            k1 = k2 = 3
            p1 = p2 = 1

        self.block = nn.Sequential(
            _conv3d(in_channels,  out_channels, k1, p1),
            _norm3d(out_channels),
            _act(),
            _conv3d(out_channels, out_channels, k2, p2),
            _norm3d(out_channels),
            _act(),
        )

    def forward(self, x): return self.block(x)

# ---------- FMU (feature merging unit) ----------
def FMU_tensor(x1, x2, mode='sub'):
    if mode == 'sum':
        return x1 + x2
    elif mode == 'sub':
        return torch.abs(x1 - x2)
    elif mode == 'cat':
        return torch.cat((x1, x2), dim=1)
    else:
        raise ValueError(f"Unexpected FMU mode: {mode}")

# ---------- BasicNet stub (matches the official style; we keep it minimal) ----------
class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        # hooks for norms/acts if you want to extend; not strictly needed here
        self.norm_kwargs = {}
        self.activation_kwargs = {}

# ---------- Down and Up modules (faithful behaviors) ----------
class Down(BasicNet):
    """
    mode: tuple(in_stream, out_stream)
      in_stream  in {'/', '2d', '3d', 'both'}; '/' means first block (no input stream split)
      out_stream in {'2d', '3d', 'both'}
    FMU: 'sum' | 'sub' | 'cat'  (feature merging when two inputs are present)
    downsample: apply pooling before block (except first)
    min_z: if D < min_z, use (1,2,2) pooling; else (2,2,2) for the 3D stream
    """
    def __init__(self, in_channels, out_channels, mode: tuple, FMU='sub',
                 downsample=True, min_z=8):
        super().__init__()
        self.mode_in, self.mode_out = mode
        self.downsample = downsample
        self.FMU_mode = FMU
        self.min_z = min_z

        if self.mode_out in ('2d','both'):
            self.CB2d = CB3d(in_channels, out_channels,
                             kSize=((1,3,3),(1,3,3)),
                             padding=(0,1,1))
        if self.mode_out in ('3d','both'):
            self.CB3d = CB3d(in_channels, out_channels,
                             kSize=(3,3),
                             padding=(1,1,1))

    def forward(self, x):
        # Downsample depending on the incoming stream configuration
        if self.downsample:
            if self.mode_in == 'both':
                x2d, x3d = x
                p2d = F.max_pool3d(x2d, kernel_size=(1,2,2), stride=(1,2,2))
                if x3d.shape[2] >= self.min_z:
                    p3d = F.max_pool3d(x3d, kernel_size=(2,2,2), stride=(2,2,2))
                else:
                    p3d = F.max_pool3d(x3d, kernel_size=(1,2,2), stride=(1,2,2))
                x = FMU_tensor(p2d, p3d, mode=self.FMU_mode)  # feature fusion
            elif self.mode_in == '2d':
                x = F.max_pool3d(x, kernel_size=(1,2,2), stride=(1,2,2))
            elif self.mode_in == '3d':
                if x.shape[2] >= self.min_z:
                    x = F.max_pool3d(x, kernel_size=(2,2,2), stride=(2,2,2))
                else:
                    x = F.max_pool3d(x, kernel_size=(1,2,2), stride=(1,2,2))
            # '/' means first node: no pooling

        # Output branches
        if self.mode_out == '2d':
            return self.CB2d(x)
        elif self.mode_out == '3d':
            return self.CB3d(x)
        elif self.mode_out == 'both':
            return self.CB2d(x), self.CB3d(x)
        else:
            raise ValueError(f"Unexpected mode_out: {self.mode_out}")

class Up(BasicNet):
    """
    mode: tuple(in_stream, out_stream)
      inputs: (x2d, xskip2d, x3d, xskip3d)
      - upsample decoder streams to skip size (trilinear)
      - FMU on both (skip2d, skip3d) and (up2d, up3d), then concat
      - run 2D and/or 3D blocks
    """
    def __init__(self, in_channels, out_channels, mode: tuple, FMU='sub'):
        super().__init__()
        self.mode_in, self.mode_out = mode
        self.FMU_mode = FMU

        if self.mode_out in ('2d','both'):
            self.CB2d = CB3d(in_channels, out_channels,
                             kSize=((1,3,3),(1,3,3)),
                             padding=(0,1,1))
        if self.mode_out in ('3d','both'):
            self.CB3d = CB3d(in_channels, out_channels,
                             kSize=(3,3),
                             padding=(1,1,1))

    def forward(self, x):
        x2d, xskip2d, x3d, xskip3d = x
        target_size = xskip2d.shape[2:]  # (D,H,W) of the corresponding skip

        up2d = F.interpolate(x2d, size=target_size, mode='trilinear', align_corners=False)
        up3d = F.interpolate(x3d, size=target_size, mode='trilinear', align_corners=False)

        cat = torch.cat([
            FMU_tensor(xskip2d, xskip3d, self.FMU_mode),   # encoder skip fusion
            FMU_tensor(up2d,    up3d,    self.FMU_mode)    # decoder fusion
        ], dim=1)

        if self.mode_out == '2d':
            return self.CB2d(cat)
        elif self.mode_out == '3d':
            return self.CB3d(cat)
        elif self.mode_out == 'both':
            return self.CB2d(cat), self.CB3d(cat)
        else:
            raise ValueError(f"Unexpected mode_out: {self.mode_out}")

# ---------- Full mesh MNet (faithful wiring) ----------
class MNetOfficialish(BasicNet):
    """
    Mirrors the official MNet module graph, channel schedule, FMU options,
    z-aware pooling, and deep supervision head layout (7 outputs when ds=True).
    """
    def __init__(self, in_channels=1, num_classes=2, kn=(32,48,64,80,96), ds=True, FMU='sub'):
        super().__init__()
        self.ds = ds
        self.num_classes = num_classes
        ch_factor = {'sum': 1, 'sub': 1, 'cat': 2}[FMU]   # 'cat' doubles channels at Up inputs

        # Row/column modules (exact names/flow mirror the reference file)
        self.down11 = Down(in_channels,    kn[0], ('/',   'both'), downsample=False)
        self.down12 = Down(kn[0],          kn[1], ('2d',  'both'))
        self.down13 = Down(kn[1],          kn[2], ('2d',  'both'))
        self.down14 = Down(kn[2],          kn[3], ('2d',  'both'))
        self.bottleneck1 = Down(kn[3],     kn[4], ('2d',  '2d'))

        self.up11 = Up(ch_factor*(kn[3]+kn[4]), kn[3], ('both','2d'),  FMU)
        self.up12 = Up(ch_factor*(kn[2]+kn[3]), kn[2], ('both','2d'),  FMU)
        self.up13 = Up(ch_factor*(kn[1]+kn[2]), kn[1], ('both','2d'),  FMU)
        self.up14 = Up(ch_factor*(kn[0]+kn[1]), kn[0], ('both','both'),FMU)

        self.down21 = Down(kn[0],          kn[1], ('3d',  'both'))
        self.down22 = Down(ch_factor*kn[1],kn[2], ('both','both'), FMU)
        self.down23 = Down(ch_factor*kn[2],kn[3], ('both','both'), FMU)
        self.bottleneck2 = Down(ch_factor*kn[3],kn[4], ('both','both'), FMU)
        self.up21 = Up(ch_factor*(kn[3]+kn[4]), kn[3], ('both','both'), FMU)
        self.up22 = Up(ch_factor*(kn[2]+kn[3]), kn[2], ('both','both'), FMU)
        self.up23 = Up(ch_factor*(kn[1]+kn[2]), kn[1], ('both','3d'),  FMU)

        self.down31 = Down(kn[1],          kn[2], ('3d',  'both'))
        self.down32 = Down(ch_factor*kn[2],kn[3], ('both','both'), FMU)
        self.bottleneck3 = Down(ch_factor*kn[3],kn[4], ('both','both'), FMU)
        self.up31 = Up(ch_factor*(kn[3]+kn[4]), kn[3], ('both','both'), FMU)
        self.up32 = Up(ch_factor*(kn[2]+kn[3]), kn[2], ('both','3d'),  FMU)

        self.down41 = Down(kn[2],          kn[3], ('3d',  'both'), FMU)
        self.bottleneck4 = Down(ch_factor*kn[3],kn[4], ('both','both'), FMU)
        self.up41 = Up(ch_factor*(kn[3]+kn[4]), kn[3], ('both','3d'),  FMU)

        self.bottleneck5 = Down(kn[3],     kn[4], ('3d',  '3d'))

        # 7 deep-supervision heads, same channel list/order as the reference
        head_chs = [kn[0], kn[1], kn[1], kn[2], kn[2], kn[3], kn[3]]
        self.outputs = nn.ModuleList([nn.Conv3d(c, num_classes, kernel_size=1, bias=False)
                                      for c in head_chs])

    def forward(self, x):
        # Row 1
        down11 = self.down11(x)           # -> (2d,3d)
        down12 = self.down12(down11[0])   # uses 2d stream
        down13 = self.down13(down12[0])
        down14 = self.down14(down13[0])
        bottleNeck1 = self.bottleneck1(down14[0])  # 2d

        # Row 2
        down21 = self.down21(down11[1])                # 3d -> both
        down22 = self.down22([down21[0], down12[1]])   # both, FMU
        down23 = self.down23([down22[0], down13[1]])
        bottleNeck2 = self.bottleneck2([down23[0], down14[1]])

        # Row 3
        down31 = self.down31(down21[1])                # 3d -> both
        down32 = self.down32([down31[0], down22[1]])
        bottleNeck3 = self.bottleneck3([down32[0], down23[1]])

        # Row 4
        down41 = self.down41(down31[1])
        bottleNeck4 = self.bottleneck4([down41[0], down32[1]])

        # Row 5
        bottleNeck5 = self.bottleneck5(down41[1])      # 3d -> 3d

        # Ups (mirror official wiring)
        up41 = self.up41([bottleNeck4[0], down41[0], bottleNeck5,  down41[1]])

        up31 = self.up31([bottleNeck3[0], down32[0], bottleNeck4[1], down32[1]])
        up32 = self.up32([up31[0],        down31[0], up41,            down31[1]])

        up21 = self.up21([bottleNeck2[0], down23[0], bottleNeck3[1],  down23[1]])
        up22 = self.up22([up21[0],        down22[0], up31[1],         down22[1]])
        up23 = self.up23([up22[0],        down21[0], up32,            down21[1]])

        up11 = self.up11([bottleNeck1,    down14[0], bottleNeck2[1],  down14[1]])
        up12 = self.up12([up11,           down13[0], up21[1],         down13[1]])
        up13 = self.up13([up12,           down12[0], up22[1],         down12[1]])
        up14 = self.up14([up13,           down11[0], up23,            down11[1]])  # -> (2d,3d)

        if self.ds:
            # Order matches the official repo: [up14 sum, up23, up13, up32, up12, up41, up11]
            features = [up14[0] + up14[1], up23, up13, up32, up12, up41, up11]
            return [self.outputs[i](features[i]) for i in range(7)]
        else:
            return self.outputs[0](up14[0] + up14[1])


if __name__ == "__main__":
    # pass

    net = MNetOfficialish(in_channels=1, num_classes=2, kn=(32,48,64,80,96), ds=True, FMU='sub')
    x = torch.randn(1,1,16,128,128)
    outs = net(x)
    assert isinstance(outs, list) and len(outs) == 7
    print([o.shape for o in outs])  # first should be [1,2,16,128,128]; others vary by scale
