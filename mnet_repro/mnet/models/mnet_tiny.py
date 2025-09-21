import torch.nn as nn
from .blocks import conv_norm_act

class MNetTiny(nn.Module):
    """Minimal placeholder network so that we can validate the pipeline.
    This is NOT the final MNet it will be replaced by the faithful implementation.
    """
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.enc = nn.Sequential(
            conv_norm_act(in_channels, 16, dim=3),
            conv_norm_act(16, 32, dim=3),
        )
        self.head = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):  # x: [B, C, D, H, W]
        z = self.enc(x)
        return self.head(z)
