from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.reproduction_mnet.basic_module import (
    CB3d,
    CB3dSeparable,
    BasicNet,
    CBzMamba,
    ZScan,
)

###############################################################################
# NEWLY ADDED VIA PROJECT REPRODUCTION

#############################################
# Fusion helpers
#############################################

def _gap(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=(2, 3, 4), keepdim=True)


class _ChannelGate(nn.Module):
    def __init__(self, channels: int, hidden: int | None = None):
        super().__init__()
        h = hidden or max(8, channels // 4)
        self.fc1 = nn.Conv3d(2 * channels, h, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(h, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.fc2.bias)  # neutral sigmoid start

    def forward(self, x2d: torch.Tensor, x3d: torch.Tensor) -> torch.Tensor:
        z = torch.cat([_gap(x2d), _gap(x3d)], dim=1)
        return torch.sigmoid(self.fc2(self.act(self.fc1(z))))


class _SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x2d: torch.Tensor, x3d: torch.Tensor) -> torch.Tensor:
        x = (x2d + x3d) * 0.5
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class FusionUnit(nn.Module):
    """
    A single entry point that implements both: classic FMU (sum/sub/cat) and
    learnable gating (channel/spatial/dual). 
    """

    def __init__(self, channels: int, fmu_mode: str = "sub", gated: str | None = None,
                 residual_blend: float = 0.0):
        super().__init__()
        assert fmu_mode in {"sum", "sub", "cat"}
        self.fmu_mode = fmu_mode
        self.gated = gated  # None | 'channel' | 'spatial' | 'dual'
        self.residual_blend = residual_blend

        self.cg = _ChannelGate(channels) if gated in {"channel", "dual"} else None
        self.sg = _SpatialGate() if gated in {"spatial", "dual"} else None

    def _plain_fmu(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.fmu_mode == "sum":
            return a + b
        if self.fmu_mode == "sub":
            return (a - b).abs()
        return torch.cat([a, b], dim=1)  # cat

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.gated is None:
            return self._plain_fmu(a, b)

        if self.gated == "channel":
            gate = self.cg(a, b)  # (N,C,1,1,1)
        elif self.gated == "spatial":
            gate = self.sg(a, b)  # (N,1,D,H,W)
        else:  # dual
            gc = self.cg(a, b)
            gs = self.sg(a, b)
            gate = torch.sigmoid(0.5 * gc + 0.5 * gs)

        y = gate * a + (1.0 - gate) * b
        if self.residual_blend:
            y = y + self.residual_blend * 0.5 * (a + b)
        return y


def _maybe_ckpt(mod: nn.Module, x: torch.Tensor, flag: bool, training: bool):
    if flag and training:
        fn = lambda t: mod(t)
        return checkpoint(fn, x)
    return mod(x)


class _BlockFactory:
    """Chooses the 3D processing block given flags."""

    def __init__(self, use_sep3d: bool, axial_vmamba: bool,
                 norm_kwargs: dict, act_kwargs: dict, axial_reduce: float = 0.5):
        self.use_sep3d = use_sep3d
        self.axial_vmamba = axial_vmamba
        self.norm_kwargs = norm_kwargs
        self.act_kwargs = act_kwargs
        self.axial_reduce = axial_reduce

    def make(self, in_ch: int, out_ch: int, for_2d: bool) -> nn.Module:
        if for_2d:
            return CB3d(
                in_channels=in_ch,
                out_channels=out_ch,
                kSize=((1, 3, 3), (1, 3, 3)),
                stride=(1, 1),
                padding=(0, 1, 1),
                norm_args=(self.norm_kwargs, self.norm_kwargs),
                activation_args=(self.act_kwargs, self.act_kwargs),
            )
        if self.axial_vmamba:
            return CBzMamba(
                in_channels=in_ch,
                out_channels=out_ch,
                reduce_ratio=self.axial_reduce,
                norm_kwargs=self.norm_kwargs,
                act_kwargs=self.act_kwargs,
            )
        Block3D = CB3dSeparable if self.use_sep3d else CB3d
        return Block3D(
            in_channels=in_ch,
            out_channels=out_ch,
            kSize=(3, 3),
            stride=(1, 1),
            padding=(1, 1, 1),
            norm_args=(self.norm_kwargs, self.norm_kwargs),
            activation_args=(self.act_kwargs, self.act_kwargs),
        )

############################################################################


############################################################################
# DOWN AND UP BLOCKS SKELETONS WERE USED FROM REPO AS REFERENCE TO ENSURE
# APPROPRIATE PATHWAYS. IMPROVEMENTS WERE MADE TO ENSURE COMPATIBILITY 
# WITH FUSION GATING AND VMAMBA BLOCKS AND MICRO-OPTIMIZATIONS ATTEMPTED 
# TO SPEED UP SIMULATIONS (EXTREMELY EXPENSIVE)

#############################################
# Core blocks 
#############################################

class DownBlock(BasicNet):
    """Downsampling stage with optional fusion and selectable 3D block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: tuple,
        *,
        FMU: str = "sub",
        downsample: bool = True,
        min_z: int = 8,
        use_sep3d: bool = False,
        use_checkpoint: bool = False,
        gated_fusion: str | None = None,
        axial_vmamba: bool = False,
        axial_reduce: float = 0.5,
    ) -> None:
        super().__init__()
        self.mode_in, self.mode_out = mode
        self.downsample = downsample
        self.min_z = min_z
        self.use_checkpoint = use_checkpoint

        factory = _BlockFactory(use_sep3d, axial_vmamba, self.norm_kwargs, self.activation_kwargs, axial_reduce)
        if self.mode_out in ("2d", "both"):
            self.cb2d = factory.make(in_channels, out_channels, for_2d=True)
        if self.mode_out in ("3d", "both"):
            self.cb3d = factory.make(in_channels, out_channels, for_2d=False)

        # fusion after pooling if two inputs are present
        self.fusion = FusionUnit(in_channels, fmu_mode=FMU, gated=gated_fusion) if (gated_fusion is not None and self.mode_in == "both") else None
        self._plain_fmu = FusionUnit(in_channels, fmu_mode=FMU, gated=None) if self.mode_in == "both" else None

    def _pool(self, x: torch.Tensor, is_3d: bool) -> torch.Tensor:
        if is_3d and x.shape[2] >= self.min_z:
            return F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        return F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        if self.downsample:
            if self.mode_in == "both":
                x2d, x3d = x
                p2d = self._pool(x2d, is_3d=False)
                p3d = self._pool(x3d, is_3d=True)
                x = (self.fusion or self._plain_fmu)(p2d, p3d)
            elif self.mode_in == "2d":
                x = self._pool(x, is_3d=False)
            elif self.mode_in == "3d":
                x = self._pool(x, is_3d=True)
            # '/' => no pooling

        if self.mode_out == "2d":
            return _maybe_ckpt(self.cb2d, x, self.use_checkpoint, self.training)
        if self.mode_out == "3d":
            return _maybe_ckpt(self.cb3d, x, self.use_checkpoint, self.training)
        # both
        return (
            _maybe_ckpt(self.cb2d, x, self.use_checkpoint, self.training),
            _maybe_ckpt(self.cb3d, x, self.use_checkpoint, self.training),
        )


class UpBlock(BasicNet):
    """Upsampling stage with fusion and optional channel reduction after concatenation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: tuple,
        *,
        FMU: str = "sub",
        use_sep3d: bool = False,
        cat_reduce: bool = False,
        use_checkpoint: bool = False,
        gated_fusion: str | None = None,
        fuse_ch: int | None = None,
        fuse_ch_up: int | None = None,
        axial_vmamba: bool = False,
        axial_reduce: float = 0.5,
    ) -> None:
        super().__init__()
        self.mode_in, self.mode_out = mode
        self.use_checkpoint = use_checkpoint

        # If FMU=cat doubles channels, allow reduction to keep widths identical
        self.reduce = _BottleneckReduce(in_channels, in_channels // 2) if (cat_reduce and FMU == "cat") else None
        in_ch_for_cb = (in_channels // 2) if self.reduce is not None else in_channels

        factory = _BlockFactory(use_sep3d, axial_vmamba, self.norm_kwargs, self.activation_kwargs, axial_reduce)
        if self.mode_out in ("2d", "both"):
            self.cb2d = factory.make(in_ch_for_cb, out_channels, for_2d=True)
        if self.mode_out in ("3d", "both"):
            self.cb3d = factory.make(in_ch_for_cb, out_channels, for_2d=False)

        # Fusion units for skip and up paths (separate sizes)
        if gated_fusion is not None:
            assert fuse_ch is not None and fuse_ch_up is not None, "UpBlock requires fuse_ch and fuse_ch_up when gated_fusion is set."
            self.fuse_skip = FusionUnit(fuse_ch, fmu_mode=FMU, gated=gated_fusion)
            self.fuse_up = FusionUnit(fuse_ch_up, fmu_mode=FMU, gated=gated_fusion)
        else:
            self.fuse_skip = FusionUnit(fuse_ch or 1, fmu_mode=FMU, gated=None)  # channels ignored if ungated
            self.fuse_up = FusionUnit(fuse_ch_up or 1, fmu_mode=FMU, gated=None)

    def forward(self, x):
        x2d, xskip2d, x3d, xskip3d = x
        tgt = xskip2d.shape[2:]

        up2d = F.interpolate(x2d, size=tgt, mode="trilinear", align_corners=False)
        up3d = F.interpolate(x3d, size=tgt, mode="trilinear", align_corners=False)

        s = self.fuse_skip(xskip2d, xskip3d)
        u = self.fuse_up(up2d, up3d)
        cat = torch.cat([s, u], dim=1)
        if self.reduce is not None:
            cat = self.reduce(cat)

        if self.mode_out == "2d":
            return _maybe_ckpt(self.cb2d, cat, self.use_checkpoint, self.training)
        if self.mode_out == "3d":
            return _maybe_ckpt(self.cb3d, cat, self.use_checkpoint, self.training)
        return (
            _maybe_ckpt(self.cb2d, cat, self.use_checkpoint, self.training),
            _maybe_ckpt(self.cb3d, cat, self.use_checkpoint, self.training),
        )

class _BottleneckReduce(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.proj(x)))

################################################################################


#############################################################
# SELF-SCRIPTED WITH GUIDANCE FROM REPOSITORY AND LLM TO ENSURE
# APPROPRIATE COMPATIBILITY WITH NNUNET INFRASTRUCTURE AND ORIGINAL 
# PAPER IMPLEMENTATION FOR CONSISTENT COMPARISON

#############################################
# Model MNet
#############################################

class MNet(SegmentationNetwork):
    # nnU-Net bookkeeping: keep identical for compatibility with original repository
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

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        kn=(32, 48, 64, 80, 96),
        ds: bool = True,
        FMU: str = "sub",
        *,
        width_mult: float = 1.0,
        use_sep3d: bool = False,
        use_checkpoint: bool = False,
        cat_reduce: bool = False,
        gated_fusion: str | None = None,
        axial_vmamba: bool = False,
        axial_reduce: float = 0.5,
    ) -> None:
        """
        additional features options:
          - width_mult, use_sep3d, use_checkpoint, cat_reduce,
            gated_fusion in {None,'channel','spatial','dual'},
            axial_vmamba/axial_reduce switches for 3D path.
        """
        super().__init__()
        self.conv_op = nn.Conv3d
        self._deep_supervision = self.do_ds = ds
        self.num_classes = num_classes

        # width scaling
        kn = tuple(max(1, int(k * width_mult)) for k in kn)

        # When using learnable gating, the fused tensor preserves channel count,
        # so we neutralize the doubling that 'cat' would have introduced.
        channel_factor = {"sum": 1, "sub": 1, "cat": 2}
        fct = 1 if gated_fusion is not None else channel_factor[FMU]

        # Stage 1 - up/down
        self.down11 = DownBlock(in_channels, kn[0], ("/", "both"), downsample=False,
                                use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, FMU=FMU)
        self.down12 = DownBlock(kn[0], kn[1], ("2d", "both"), use_sep3d=use_sep3d, 
                            use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, FMU=FMU)
        self.down13 = DownBlock(kn[1], kn[2], ("2d", "both"), use_sep3d=use_sep3d, 
                            use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, FMU=FMU)
        self.down14 = DownBlock(kn[2], kn[3], ("2d", "both"), use_sep3d=use_sep3d, 
                            use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, FMU=FMU)
        self.bottleneck1 = DownBlock(kn[3], kn[4], ("2d", "2d"), use_sep3d=use_sep3d, 
                            use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, FMU=FMU)

        self.up11 = UpBlock(fct * (kn[3] + kn[4]), kn[3], ("both", "2d"), FMU=FMU,
                            use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint,
                            gated_fusion=gated_fusion, fuse_ch=kn[3], fuse_ch_up=kn[4])
        self.up12 = UpBlock(fct * (kn[2] + kn[3]), kn[2], ("both", "2d"), FMU=FMU,
                            use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint,
                            gated_fusion=gated_fusion, fuse_ch=kn[2], fuse_ch_up=kn[3])
        self.up13 = UpBlock(fct * (kn[1] + kn[2]), kn[1], ("both", "2d"), FMU=FMU,
                            use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint,
                            gated_fusion=gated_fusion, fuse_ch=kn[1], fuse_ch_up=kn[2])
        self.up14 = UpBlock(fct * (kn[0] + kn[1]), kn[0], ("both", "both"), FMU=FMU,
                            use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint,
                            gated_fusion=gated_fusion, fuse_ch=kn[0], fuse_ch_up=kn[1])

        # Stage 2 - up/down
        self.down21 = DownBlock(kn[0], kn[1], ("3d", "both"), FMU=FMU,
                                use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion,
                                axial_vmamba=False, axial_reduce=0.5)
        self.down22 = DownBlock(fct * kn[1], kn[2], ("both", "both"), FMU=FMU,
                                use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion,
                                axial_vmamba=False, axial_reduce=0.5)
        self.down23 = DownBlock(fct * kn[2], kn[3], ("both", "both"), FMU=FMU,
                                use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion,
                                axial_vmamba=False, axial_reduce=0.5)
        self.bottleneck2 = DownBlock(fct * kn[3], kn[4], ("both", "both"), FMU=FMU,
                                     use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion,
                                     axial_vmamba=False, axial_reduce=0.5)
        self.up21 = UpBlock(fct * (kn[3] + kn[4]), kn[3], ("both", "both"), FMU=FMU,
                            use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint,
                            gated_fusion=gated_fusion, fuse_ch=kn[3], fuse_ch_up=kn[4], axial_vmamba=False, axial_reduce=0.5)
        self.up22 = UpBlock(fct * (kn[2] + kn[3]), kn[2], ("both", "both"), FMU=FMU,
                            use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint,
                            gated_fusion=gated_fusion, fuse_ch=kn[2], fuse_ch_up=kn[3], axial_vmamba=False, axial_reduce=0.5)
        self.up23 = UpBlock(fct * (kn[1] + kn[2]), kn[1], ("both", "3d"), FMU=FMU,
                            use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint,
                            gated_fusion=gated_fusion, fuse_ch=kn[1], fuse_ch_up=kn[2], axial_vmamba=False, axial_reduce=0.5)

        # Stage 3 - up/down
        self.down31 = DownBlock(kn[1], kn[2], ("3d", "both"), use_sep3d=use_sep3d, 
                            use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, FMU=FMU)
        self.down32 = DownBlock(fct * kn[2], kn[3], ("both", "both"), use_sep3d=use_sep3d, 
                            use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, FMU=FMU)
        self.bottleneck3 = DownBlock(fct * kn[3], kn[4], ("both", "both"), use_sep3d=use_sep3d, 
                            use_checkpoint=use_checkpoint, gated_fusion=gated_fusion, FMU=FMU, axial_vmamba=axial_vmamba, axial_reduce=axial_reduce)
        self.up31 = UpBlock(fct * (kn[3] + kn[4]), kn[3], ("both", "both"), FMU=FMU,
                            use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint,
                            gated_fusion=gated_fusion, fuse_ch=kn[3], fuse_ch_up=kn[4])
        self.up32 = UpBlock(fct * (kn[2] + kn[3]), kn[2], ("both", "3d"), FMU=FMU,
                            use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint,
                            gated_fusion=gated_fusion, fuse_ch=kn[2], fuse_ch_up=kn[3])

        # Stage 4 - up/down
        self.down41 = DownBlock(kn[2], kn[3], ("3d", "both"), FMU=FMU, use_sep3d=use_sep3d, use_checkpoint=use_checkpoint, gated_fusion=gated_fusion)
        self.bottleneck4 = DownBlock(fct * kn[3], kn[4], ("both", "both"), FMU=FMU, use_sep3d=use_sep3d, use_checkpoint=use_checkpoint,
                                     gated_fusion=gated_fusion, axial_vmamba=axial_vmamba, axial_reduce=axial_reduce)
        self.up41 = UpBlock(fct * (kn[3] + kn[4]), kn[3], ("both", "3d"), FMU=FMU,
                            use_sep3d=use_sep3d, cat_reduce=cat_reduce, use_checkpoint=use_checkpoint,
                            gated_fusion=gated_fusion, fuse_ch=kn[3], fuse_ch_up=kn[4])

        self.bottleneck5 = DownBlock(kn[3], kn[4], ("3d", "3d"), use_sep3d=use_sep3d, use_checkpoint=use_checkpoint,
                                     gated_fusion=gated_fusion, FMU=FMU, axial_vmamba=axial_vmamba, axial_reduce=axial_reduce)

        # output heads identical to original repository
        self.outputs = nn.ModuleList(
            [
                nn.Conv3d(c, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
                for c in [kn[0], kn[1], kn[1], kn[2], kn[2], kn[3], kn[3]]
            ]
        )

        self.inference_apply_nonlin = None

    @property
    def deep_supervision(self):
        return self._deep_supervision

    # follows the same structure to ensure consistency with the original repository
    def forward(self, x: torch.Tensor):
        # 1
        d11 = self.down11(x)
        d12 = self.down12(d11[0])
        d13 = self.down13(d12[0])
        d14 = self.down14(d13[0])
        b1 = self.bottleneck1(d14[0])

        # 2
        d21 = self.down21(d11[1])
        d22 = self.down22([d21[0], d12[1]])
        d23 = self.down23([d22[0], d13[1]])
        b2 = self.bottleneck2([d23[0], d14[1]])

        # 3
        d31 = self.down31(d21[1])
        d32 = self.down32([d31[0], d22[1]])
        b3 = self.bottleneck3([d32[0], d23[1]])

        # 4
        d41 = self.down41(d31[1])
        b4 = self.bottleneck4([d41[0], d32[1]])
        b5 = self.bottleneck5(d41[1])

        # reverse 4
        u41 = self.up41([b4[0], d41[0], b5, d41[1]])

        # reverse 3
        u31 = self.up31([b3[0], d32[0], b4[1], d32[1]])
        u32 = self.up32([u31[0], d31[0], u41, d31[1]])

        # reverse 2
        u21 = self.up21([b2[0], d23[0], b3[1], d23[1]])
        u22 = self.up22([u21[0], d22[0], u31[1], d22[1]])
        u23 = self.up23([u22[0], d21[0], u32, d21[1]])

        # reverse 1
        u11 = self.up11([b1, d14[0], b2[1], d14[1]])
        u12 = self.up12([u11, d13[0], u21[1], d13[1]])
        u13 = self.up13([u12, d12[0], u22[1], d12[1]])
        u14 = self.up14([u13, d11[0], u23, d11[1]])

        if self._deep_supervision and self.do_ds:
            feats = [u14[0] + u14[1], u23, u13, u32, u12, u41, u11]
            return tuple(self.outputs[i](feats[i]) for i in range(7))
        return self.outputs[0](u14[0] + u14[1])


if __name__ == "__main__":
    net = MNet(1, 3, kn=(16, 24, 32, 40, 48), ds=True, FMU="sub",
               width_mult=1.0, use_sep3d=False, use_checkpoint=False, cat_reduce=False)
    x = torch.randn(1, 1, 19, 128, 128)
    y = net(x)
    print([t.shape for t in y] if isinstance(y, tuple) else y.shape)
