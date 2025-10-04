from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.reproduction_mnet.basic_module import CB3d, CB3dSeparable, BasicNet

# ------------------------
# Utilities / helpers
# ------------------------

# ----- Gated fusion modules -----
def _gap(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=(2, 3, 4), keepdim=True)  # (N,C,1,1,1)

class ChannelGate(nn.Module):
    def __init__(self, channels: int, hidden: int = None):
        super().__init__()
        h = hidden or max(8, channels // 4)
        self.fc1 = nn.Conv3d(2 * channels, h, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(h, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.fc2.bias)  # sigmoid(0)=0.5 -> neutral start

    def forward(self, x2d: torch.Tensor, x3d: torch.Tensor) -> torch.Tensor:
        z = torch.cat([_gap(x2d), _gap(x3d)], dim=1)  # (N,2C,1,1,1)
        g = torch.sigmoid(self.fc2(self.act(self.fc1(z))))  # (N,C,1,1,1)
        return g

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x2d: torch.Tensor, x3d: torch.Tensor) -> torch.Tensor:
        x = 0.5 * (x2d + x3d)
        avg = x.mean(dim=1, keepdim=True)        # (N,1,D,H,W)
        mx  = x.amax(dim=1, keepdim=True)        # (N,1,D,H,W)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # (N,1,D,H,W)

class GatedFMU(nn.Module):
    """
    mode: 'channel' | 'spatial' | 'dual'
    Returns fused tensor with SAME channels as inputs.
    """
    def __init__(self, channels: int, mode: str = "channel", residual_blend: float = 0.0):
        super().__init__()
        assert mode in {"channel", "spatial", "dual"}
        self.mode = mode
        self.residual_blend = residual_blend
        self.cg = ChannelGate(channels) if mode in {"channel", "dual"} else None
        self.sg = SpatialGate()         if mode in {"spatial", "dual"} else None

    def forward(self, x2d: torch.Tensor, x3d: torch.Tensor) -> torch.Tensor:
        if self.mode == "channel":
            g = self.cg(x2d, x3d)                         # (N,C,1,1,1)
        elif self.mode == "spatial":
            g = self.sg(x2d, x3d)                         # (N,1,D,H,W)
        else:
            gc = self.cg(x2d, x3d)                        # (N,C,1,1,1)
            gs = self.sg(x2d, x3d)                        # (N,1,D,H,W)
            g = torch.sigmoid(0.5 * gc + 0.5 * gs)        # broadcast add

        y = g * x2d + (1.0 - g) * x3d
        if self.residual_blend > 0:
            y = y + self.residual_blend * 0.5 * (x2d + x3d)
        return y

def FMU(x1: torch.Tensor, x2: torch.Tensor, mode: str = 'sub') -> torch.Tensor:
    if mode == 'sum':
        return torch.add(x1, x2)
    elif mode == 'sub':
        return torch.abs(x1 - x2)
    elif mode == 'cat':
        return torch.cat((x1, x2), dim=1)
    else:
        raise ValueError(f'Unexpected FMU mode: {mode}')

class Bottleneck1x1(nn.Module):
    """Cheap channel reducer used when FMU='cat' (doubles channels)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.proj(x)))

def ckpt_if(module: nn.Module, x: torch.Tensor, use_ckpt: bool, training: bool):
    # checkpoint only on training to avoid inference overhead
    if use_ckpt and training:
        def fn(t): return module(t)
        return checkpoint(fn, x)
    return module(x)

# ------------------------
# Core blocks
# ------------------------

class Down(BasicNet):
    """
    Down block with anisotropy-aware pooling and selectable 3D block type.
    """
    def __init__(self, in_channels, out_channels, mode: tuple, FMU='sub', downsample=True, min_z=8,
                 use_sep3d: bool = False, use_checkpoint: bool = False, gated_fusion: str = None,
                 
                 # additional args for ZScan via Mamba-1D:
                 axial_vmamba: bool = False, axial_reduce: float = 0.5,):
        
        super().__init__()
        self.mode_in, self.mode_out = mode
        self.downsample = downsample
        self.FMU = FMU
        self.min_z = min_z
        self.use_checkpoint = use_checkpoint
        norm_args = (self.norm_kwargs, self.norm_kwargs)
        activation_args = (self.activation_kwargs, self.activation_kwargs)
        self.fuse = None

        if self.mode_out in ('2d', 'both'):
            self.CB2d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=((1, 3, 3), (1, 3, 3)), stride=(1, 1), padding=(0, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)

        if self.mode_out in ('3d', 'both'):
            if axial_vmamba:
                self.CB3d = CBzMamba(in_channels=in_channels, out_channels=out_channels,
                                    reduce_ratio=axial_reduce,
                                    norm_kwargs=self.norm_kwargs, act_kwargs=self.activation_kwargs)
            else:
                Block3D = CB3dSeparable if use_sep3d else CB3d
                self.CB3d = Block3D(in_channels=in_channels, out_channels=out_channels,
                                    kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1),
                                    norm_args=norm_args, activation_args=activation_args)

        if gated_fusion is not None and self.mode_in == 'both': # in ('both', '/'):
            # we fuse AFTER pooling, before convs; fused tensor feeds both heads
            self.fuse = GatedFMU(in_channels, mode=gated_fusion)  # in_channels is the per-level width

    def forward(self, x):
        if self.downsample:
            if self.mode_in == 'both':
                x2d, x3d = x
                p2d = F.max_pool3d(x2d, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                if x3d.shape[2] >= self.min_z:
                    p3d = F.max_pool3d(x3d, kernel_size=(2, 2, 2), stride=(2, 2, 2))
                else:
                    p3d = F.max_pool3d(x3d, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                # x = FMU(p2d, p3d, mode=self.FMU)
                x = self.fuse(p2d, p3d) if self.fuse is not None else FMU(p2d, p3d, mode=self.FMU)

            elif self.mode_in == '2d':
                x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

            elif self.mode_in == '3d':
                if x.shape[2] >= self.min_z:
                    x = F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
                else:
                    x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            # '/' does no pooling

        if self.mode_out == '2d':
            return ckpt_if(self.CB2d, x, self.use_checkpoint, self.training)
        elif self.mode_out == '3d':
            return ckpt_if(self.CB3d, x, self.use_checkpoint, self.training)
        elif self.mode_out == 'both':
            return (ckpt_if(self.CB2d, x, self.use_checkpoint, self.training),
                    ckpt_if(self.CB3d, x, self.use_checkpoint, self.training))


class Up(BasicNet):
    """
    Up block with FMU fusion; optional post-FMU 1x1x1 reduction when FMU='cat'.
    """
    def __init__(self, in_channels, out_channels, mode: tuple, FMU='sub',
                 use_sep3d: bool = False, cat_reduce: bool = False, use_checkpoint: bool = False,
                 gated_fusion: str = None, fuse_ch: int = None, fuse_ch_up: int = None,
                 
                 # additional args for ZScan via Mamba-1D:
                                  axial_vmamba: bool = False, axial_reduce: float = 0.5):
        
        super().__init__()
        self.mode_in, self.mode_out = mode
        self.FMU = FMU
        self.cat_reduce = cat_reduce and (FMU == 'cat')
        self.use_checkpoint = use_checkpoint

        norm_args = (self.norm_kwargs, self.norm_kwargs)
        activation_args = (self.activation_kwargs, self.activation_kwargs)

        # Optional reducer if FMU='cat' doubled channels
        self.reduce = Bottleneck1x1(in_channels, in_channels // 2) if self.cat_reduce else None
        in_ch_for_cb = (in_channels // 2) if self.cat_reduce else in_channels

        if self.mode_out in ('2d', 'both'):
            self.CB2d = CB3d(in_channels=in_ch_for_cb, out_channels=out_channels,
                             kSize=((1, 3, 3), (1, 3, 3)), stride=(1, 1), padding=(0, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)

        if self.mode_out in ('3d', 'both'):
            if axial_vmamba:
                self.CB3d = CBzMamba(in_channels=in_ch_for_cb, out_channels=out_channels,
                                    reduce_ratio=axial_reduce,
                                    norm_kwargs=self.norm_kwargs, act_kwargs=self.activation_kwargs)
            else:
                Block3D = CB3dSeparable if use_sep3d else CB3d
                self.CB3d = Block3D(in_channels=in_ch_for_cb, out_channels=out_channels,
                                    kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1),
                                    norm_args=norm_args, activation_args=activation_args)

        if gated_fusion is not None:
            assert fuse_ch is not None, "Up needs fuse_ch (= per-stream channels at this level)"
            self.fuse = GatedFMU(fuse_ch, mode=gated_fusion)

        self.fuse_skip = None
        self.fuse_up   = None
        if gated_fusion is not None:
            assert fuse_ch is not None, "Up needs fuse_ch (= skip channels at this level)"
            assert fuse_ch_up is not None, "Up needs fuse_ch_up (= up channels from deeper level)"
            self.fuse_skip = GatedFMU(fuse_ch,    mode=gated_fusion)
            self.fuse_up   = GatedFMU(fuse_ch_up, mode=gated_fusion)

    def forward(self, x):
        x2d, xskip2d, x3d, xskip3d = x
        tarSize = xskip2d.shape[2:]

        up2d = F.interpolate(x2d, size=tarSize, mode='trilinear', align_corners=False)
        up3d = F.interpolate(x3d, size=tarSize, mode='trilinear', align_corners=False)
        
        # fuse skips with gate sized for C_skip; ups with gate sized for C_up
        if self.fuse_skip is not None and self.fuse_up is not None:
            fused_skips = self.fuse_skip(xskip2d, xskip3d)  # (N, C_skip, D,H,W)
            fused_ups   = self.fuse_up(up2d, up3d)          # (N, C_up,   D,H,W)
        else:
            fused_skips = FMU(xskip2d, xskip3d, self.FMU)
            fused_ups   = FMU(up2d,   up3d,   self.FMU)

        cat = torch.cat([fused_skips, fused_ups], dim=1)
        # cat = torch.cat([FMU(xskip2d, xskip3d, self.FMU), FMU(up2d, up3d, self.FMU)], dim=1)
        if self.reduce is not None:
            cat = self.reduce(cat)

        if self.mode_out == '2d':
            return ckpt_if(self.CB2d, cat, self.use_checkpoint, self.training)
        elif self.mode_out == '3d':
            return ckpt_if(self.CB3d, cat, self.use_checkpoint, self.training)
        elif self.mode_out == 'both':
            return (ckpt_if(self.CB2d, cat, self.use_checkpoint, self.training),
                    ckpt_if(self.CB3d, cat, self.use_checkpoint, self.training))


class MNet(SegmentationNetwork):
    # nnU-Net bookkeeping
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000

    def __init__(self, in_channels, num_classes, kn=(32, 48, 64, 80, 96), ds=True, FMU='sub',
                 width_mult: float = 1.0, use_sep3d: bool = False, use_checkpoint: bool = False, 
                 cat_reduce: bool = False, gated_fusion: str = None):
        """
        Efficiency knobs (all default off):
          - width_mult: scales channel counts uniformly
          - use_sep3d: depthwise-separable 3D convs for 3D path
          - use_checkpoint: gradient checkpoint CB blocks (VRAM saver)
          - cat_reduce: add 1x1x1 reducer after FMU='cat' fusion
        """
        super().__init__()
        self.conv_op = nn.Conv3d
        self._deep_supervision = self.do_ds = ds
        self.num_classes = num_classes

        kn = tuple(max(1, int(k * width_mult)) for k in kn)
        channel_factor = {'sum': 1, 'sub': 1, 'cat': 2}
        # fct = channel_factor[FMU]
        fct = 1 if gated_fusion is not None else channel_factor[FMU]

        # Stage 1
        self.down11 = Down(in_channels, kn[0], ('/', 'both'), downsample=False, use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion)
        self.down12 = Down(kn[0], kn[1], ('2d', 'both'), use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion)
        self.down13 = Down(kn[1], kn[2], ('2d', 'both'), use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion)
        self.down14 = Down(kn[2], kn[3], ('2d', 'both'), use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion)
        self.bottleneck1 = Down(kn[3], kn[4], ('2d', '2d'), use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion)
        self.up11 = Up(fct * (kn[3] + kn[4]), kn[3], ('both', '2d'), FMU, use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, fuse_ch=kn[3], fuse_ch_up=kn[4])
        self.up12 = Up(fct * (kn[2] + kn[3]), kn[2], ('both', '2d'), FMU, use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, fuse_ch=kn[2], fuse_ch_up=kn[3])
        self.up13 = Up(fct * (kn[1] + kn[2]), kn[1], ('both', '2d'), FMU, use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, fuse_ch=kn[1], fuse_ch_up=kn[2])
        self.up14 = Up(fct * (kn[0] + kn[1]), kn[0], ('both', 'both'), FMU, use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, fuse_ch=kn[0], fuse_ch_up=kn[1])

        # Stage 2
        # Axial VMamba in first Down of 2.5D path
        self.down21 = Down(kn[0], kn[1], ('3d', 'both'), FMU, use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, axial_vmamba=True, axial_reduce=0.5
        self.down22 = Down(fct * kn[1], kn[2], ('both', 'both'), FMU, use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, axial_vmamba=True, axial_reduce=0.5)
        self.down23 = Down(fct * kn[2], kn[3], ('both', 'both'), FMU, use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, axial_vmamba=True, axial_reduce=0.5)
        self.bottleneck2 = Down(fct * kn[3], kn[4], ('both', 'both'), FMU, use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, axial_vmamba=True)
        self.up21 = Up(fct * (kn[3] + kn[4]), kn[3], ('both', 'both'), FMU, use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, fuse_ch=kn[3], fuse_ch_up=kn[4], axial_vmamba=True, axial_reduce=0.5)
        self.up22 = Up(fct * (kn[2] + kn[3]), kn[2], ('both', 'both'), FMU, use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, fuse_ch=kn[2], fuse_ch_up=kn[3], axial_vmamba=True, axial_reduce=0.5)
        self.up23 = Up(fct * (kn[1] + kn[2]), kn[1], ('both', '3d'), FMU, use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, fuse_ch=kn[1], fuse_ch_up=kn[2], axial_vmamba=True, axial_reduce=0.5)

        # Stage 3
        self.down31 = Down(kn[1], kn[2], ('3d', 'both'), use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion)
        self.down32 = Down(fct * kn[2], kn[3], ('both', 'both'), FMU, use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion)
        self.bottleneck3 = Down(fct * kn[3], kn[4], ('both', 'both'), FMU, use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion)
        self.up31 = Up(fct * (kn[3] + kn[4]), kn[3], ('both', 'both'), FMU, use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, fuse_ch=kn[3], fuse_ch_up=kn[4])
        self.up32 = Up(fct * (kn[2] + kn[3]), kn[2], ('both', '3d'), FMU, use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, fuse_ch=kn[2], fuse_ch_up=kn[3])

        # Stage 4
        self.down41 = Down(kn[2], kn[3], ('3d', 'both'), FMU, use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion)
        self.bottleneck4 = Down(fct * kn[3], kn[4], ('both', 'both'), FMU, use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion)
        self.up41 = Up(fct * (kn[3] + kn[4]), kn[3], ('both', '3d'), FMU, use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, fuse_ch=kn[3], fuse_ch_up=kn[4])

        self.bottleneck5 = Down(kn[3], kn[4], ('3d', '3d'), use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion)

        # Deep supervision heads
        self.outputs = nn.ModuleList(
            [nn.Conv3d(c, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
             for c in [kn[0], kn[1], kn[1], kn[2], kn[2], kn[3], kn[3]]]
        )

        self.inference_apply_nonlin = None

    @property
    def deep_supervision(self):
        return self._deep_supervision

    def forward(self, x: torch.Tensor):
        # 1
        down11 = self.down11(x)
        down12 = self.down12(down11[0])
        down13 = self.down13(down12[0])
        down14 = self.down14(down13[0])
        bottleNeck1 = self.bottleneck1(down14[0])

        # 2
        down21 = self.down21(down11[1])
        down22 = self.down22([down21[0], down12[1]])
        down23 = self.down23([down22[0], down13[1]])
        bottleNeck2 = self.bottleneck2([down23[0], down14[1]])

        # 3
        down31 = self.down31(down21[1])
        down32 = self.down32([down31[0], down22[1]])
        bottleNeck3 = self.bottleneck3([down32[0], down23[1]])

        # 4
        down41 = self.down41(down31[1])
        bottleNeck4 = self.bottleneck4([down41[0], down32[1]])

        bottleNeck5 = self.bottleneck5(down41[1])

        up41 = self.up41([bottleNeck4[0], down41[0], bottleNeck5, down41[1]])

        up31 = self.up31([bottleNeck3[0], down32[0], bottleNeck4[1], down32[1]])
        up32 = self.up32([up31[0], down31[0], up41, down31[1]])

        up21 = self.up21([bottleNeck2[0], down23[0], bottleNeck3[1], down23[1]])
        up22 = self.up22([up21[0], down22[0], up31[1], down22[1]])
        up23 = self.up23([up22[0], down21[0], up32, down21[1]])

        up11 = self.up11([bottleNeck1, down14[0], bottleNeck2[1], down14[1]])
        up12 = self.up12([up11, down13[0], up21[1], down13[1]])
        up13 = self.up13([up12, down12[0], up22[1], down12[1]])
        up14 = self.up14([up13, down11[0], up23, down11[1]])

        if self._deep_supervision and self.do_ds:
            features = [up14[0] + up14[1], up23, up13, up32, up12, up41, up11]
            return tuple(self.outputs[i](features[i]) for i in range(7))
        else:
            return self.outputs[0](up14[0] + up14[1])


if __name__ == '__main__':
    net = MNet(1, 3, kn=(16, 24, 32, 40, 48), ds=True, FMU='sub',
               width_mult=1.0, use_sep3d=False, use_checkpoint=False, cat_reduce=False)
    x = torch.randn(1, 1, 19, 128, 128)
    y = net(x)
    print([t.shape for t in y] if isinstance(y, tuple) else y.shape)
