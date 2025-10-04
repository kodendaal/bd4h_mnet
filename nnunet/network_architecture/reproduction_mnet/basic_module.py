from torch import nn
import torch

# optional import for Mamba 1D acceleration
try:
    # If you have mamba-ssm (or a Vision Mamba 1D), import it here.
    from mamba_ssm import Mamba as Mamba1D  # example API; change to your impl
except Exception:
    Mamba1D = None
    
# ------------------------
# Core blocks (with micro-optimizations)
# ------------------------

class CNA3d(nn.Module):
    """
    Conv3d (+ optional InstanceNorm3d + optional LeakyReLU)
    - Disables conv bias when normalization is present (tiny free win).
    """
    def __init__(self, in_channels, out_channels, kSize, stride, padding=(1, 1, 1),
                 bias=True, norm_args=None, activation_args=None):
        super().__init__()
        use_norm = norm_args is not None
        conv_bias = False if use_norm else bias

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kSize,
                              stride=stride, padding=padding, bias=conv_bias)
        self.norm = nn.InstanceNorm3d(out_channels, **norm_args) if use_norm else None
        self.activation = nn.LeakyReLU(**activation_args) if activation_args is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None: x = self.norm(x)
        if self.activation is not None: x = self.activation(x)
        return x


class CB3d(nn.Module):
    """
    3D conv block = CNA3d -> CNA3d
    kSize can be like (3,3) or ((1,3,3),(1,3,3)) for 2.5D.
    """
    def __init__(self, in_channels, out_channels, kSize=(3, 3), stride=(1, 1),
                 padding=(1, 1, 1), bias=True, norm_args=(None, None), activation_args=(None, None)):
        super().__init__()
        self.conv1 = CNA3d(in_channels, out_channels, kSize=kSize[0], stride=stride[0],
                           padding=padding, bias=bias, norm_args=norm_args[0], activation_args=activation_args[0])
        self.conv2 = CNA3d(out_channels, out_channels, kSize=kSize[1], stride=stride[1],
                           padding=padding, bias=bias, norm_args=norm_args[1], activation_args=activation_args[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# ------------------------
# Optional: Depthwise-separable variant for 3D path (opt-in)
# ------------------------

class CB3dSeparable(nn.Module):
    """
    Depthwise (groups=channels) 3D + pointwise 1x1x1, twice.
    Only recommended for the true 3D stream; keep 2.5D path standard.
    """
    def __init__(self, in_channels, out_channels, kSize=(3, 3), stride=(1, 1),
                 padding=(1, 1, 1), norm_args=(None, None), activation_args=(None, None)):
        super().__init__()
        # depthwise -> pointwise
        self.dw1 = nn.Conv3d(in_channels, in_channels, kernel_size=kSize[0], stride=stride[0],
                             padding=padding, groups=in_channels, bias=False)
        self.n1 = nn.InstanceNorm3d(in_channels, **(norm_args[0] if norm_args[0] is not None else {'affine': True}))
        self.a1 = nn.LeakyReLU(**(activation_args[0] if activation_args[0] is not None else {'negative_slope': 1e-2, 'inplace': True}))
        self.pw1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

        self.dw2 = nn.Conv3d(out_channels, out_channels, kernel_size=kSize[1], stride=stride[1],
                             padding=padding, groups=out_channels, bias=False)
        self.n2 = nn.InstanceNorm3d(out_channels, **(norm_args[1] if norm_args[1] is not None else {'affine': True}))
        self.a2 = nn.LeakyReLU(**(activation_args[1] if activation_args[1] is not None else {'negative_slope': 1e-2, 'inplace': True}))
        self.pw2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pw1(self.a1(self.n1(self.dw1(x))))
        x = self.pw2(self.a2(self.n2(self.dw2(x))))
        return x

# ------------------------
# Optional: Mamba 1D variant for 2.5D path (opt-in)
# ------------------------
class ZScan(nn.Module):
    """
    Z-axis sequence modeling at each (h,w).
    If Mamba1D is present -> use it; else fall back to separable Conv along Z.
    Input/Output: (N, C, D, H, W)
    """
    def __init__(self, channels: int, k_fallback: int = 5):
        super().__init__()
        self.use_mamba = Mamba1D is not None
        if self.use_mamba:
            self.block = Mamba1D(d_model=channels)  # adjust args to your Mamba-1D
        else:
            # depthwise conv only along Z as a lightweight proxy
            pad = k_fallback // 2
            self.block = nn.Conv3d(channels, channels, kernel_size=(k_fallback, 1, 1),
                                   padding=(pad, 0, 0), groups=channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_mamba:
            N, C, D, H, W = x.shape
            x_perm = x.permute(0, 3, 4, 2, 1).contiguous()    # (N,H,W,D,C)
            seq = x_perm.view(N * H * W, D, C)                # (B*, L=D, C)
            yseq = self.block(seq)                            # (B*, L, C)
            y = yseq.view(N, H, W, D, C).permute(0, 4, 3, 1, 2).contiguous()
            return y
        else:
            return self.block(x)


class CBzMamba(nn.Module):
    """
    Axial-hybrid inter-slice branch: 1x1 reduce -> ZScan(Mamba/conv) -> 1x1 expand (+ norm/act)
    Drop-in replacement for your CB3d in the '3d' path.
    """
    def __init__(self, in_channels, out_channels, reduce_ratio: float = 0.5,
                 norm_kwargs={'affine': True}, act_kwargs={'negative_slope': 1e-2, 'inplace': True}):
        super().__init__()
        mid = max(8, int(in_channels * reduce_ratio))
        self.pre = nn.Conv3d(in_channels, mid, kernel_size=1, bias=False)
        self.n1  = nn.InstanceNorm3d(mid, **norm_kwargs)
        self.a1  = nn.LeakyReLU(**act_kwargs)
        self.zssm = ZScan(mid)                     # <-- axial Mamba
        self.post = nn.Conv3d(mid, out_channels, kernel_size=1, bias=False)
        self.n2  = nn.InstanceNorm3d(out_channels, **norm_kwargs)
        self.a2  = nn.LeakyReLU(**act_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a1(self.n1(self.pre(x)))
        x = self.zssm(x)
        x = self.a2(self.n2(self.post(x)))
        return x
    
# ------------------------
# Base class
# ------------------------

class BasicNet(nn.Module):
    norm_kwargs = {'affine': True}
    activation_kwargs = {'negative_slope': 1e-2, 'inplace': True}

    def __init__(self):
        super().__init__()

    def parameter_count(self):
        print("Model has {:.2f}M parameters".format(sum(x.numel() for x in self.parameters()) / 1e6))
