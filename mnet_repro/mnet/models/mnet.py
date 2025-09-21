import torch
import torch.nn as nn
import torch.nn.functional as F

def norm_act(ch, dim=3, leak=0.1):
    if dim==3:
        n = nn.InstanceNorm3d(ch, affine=True)
    else:
        n = nn.InstanceNorm2d(ch, affine=True)
    return nn.Sequential(n, nn.LeakyReLU(leak, inplace=True))

def conv_block_3d(in_ch, out_ch, k1=(1,3,3), k2=(3,3,3), leak=0.1):
    p1 = tuple((kk-1)//2 for kk in k1)
    p2 = tuple((kk-1)//2 for kk in k2)
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, k1, padding=p1, bias=False),
        nn.InstanceNorm3d(out_ch, affine=True),
        nn.LeakyReLU(leak, inplace=True),
        nn.Conv3d(out_ch, out_ch, k2, padding=p2, bias=False),
        nn.InstanceNorm3d(out_ch, affine=True),
        nn.LeakyReLU(leak, inplace=True),
    )

class Block2D(nn.Module):
    def __init__(self, in_ch, out_ch, leak=0.1):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(leak, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(leak, inplace=True),
        )
    def forward(self, x):  # x: [B,C,D,H,W]
        B,C,D,H,W = x.shape
        x2 = x.permute(0,2,1,3,4).contiguous().view(B*D, C, H, W)
        x2 = self.c1(x2)
        x2 = x2.view(B, D, -1, H, W).permute(0,2,1,3,4).contiguous()
        return x2

class DownHW(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
    def forward(self, x):
        return self.pool(x)


class FMU(nn.Module):
    """
    Feature Matching Unit (paper-inspired):
    Upsample 2D and 3D features to a common target size, then abs-diff.
    Optionally fuse encoder skips via abs-diff and concatenate.
    """
    def __init__(self, fuse_skip=False):
        super().__init__()
        self.fuse_skip = fuse_skip

    def forward(self, f2d, f3d, target_size, skip2d=None, skip3d=None):
        """
        f2d, f3d: [B,C,D,H,W] at the same scale (may differ from target_size)
        target_size: (D,H,W) to match (e.g., the current decoder 'up' tensor size)
        skip2d, skip3d: optional encoder skip features (same size as f2d/f3d);
                        they will be upsampled to target_size if provided.
        """
        # Trilinear upsample to the *same* target spatial size.
        f2d_up = F.interpolate(f2d, size=target_size, mode="trilinear", align_corners=False)
        f3d_up = F.interpolate(f3d, size=target_size, mode="trilinear", align_corners=False)
        fm = torch.abs(f2d_up - f3d_up)

        if self.fuse_skip and skip2d is not None and skip3d is not None:
            s2 = F.interpolate(skip2d, size=target_size, mode="trilinear", align_corners=False)
            s3 = F.interpolate(skip3d, size=target_size, mode="trilinear", align_corners=False)
            fm_skip = torch.abs(s2 - s3)
            return torch.cat([fm, fm_skip], dim=1)

        return fm


class MNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, base_ch=32, depth=4,
                 deep_supervision=True, ds_heads=3):
        """
        ds_heads: number of auxiliary heads from coarse->fine (<= depth)
        """
        super().__init__()
        chs = [base_ch*(i+1) for i in range(depth)]  # 32,64,96,128 ... (moderate growth)

        # encoder stage 0
        self.enc3d_0 = conv_block_3d(in_channels, chs[0])
        self.enc2d_0 = Block2D(in_channels, chs[0])
        self.down0   = DownHW()
        self.fuse0   = nn.Conv3d(chs[0]*2, chs[0], 1, bias=False)  # simple concat+1x1 to seed

        # encoder stages 1..depth-1
        self.enc3d, self.enc2d, self.downs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(1, depth):
            self.enc3d.append(conv_block_3d(chs[i-1], chs[i]))
            self.enc2d.append(Block2D(chs[i-1], chs[i]))
            self.downs.append(DownHW())

        # bottleneck
        self.bott = conv_block_3d(chs[-1], chs[-1]*2, k1=(3,3,3), k2=(3,3,3))

        # decoder
        dec_chs = list(reversed(chs))  # skip sizes
        self.dec_convs = nn.ModuleList()
        self.fmus = nn.ModuleList()
        in_ch = chs[-1]*2
        for skip_ch in dec_chs:
            # upsample by (1,2,2) then fuse with FMU(abs-diff)
            self.fmus.append(FMU(fuse_skip=True))
            # after FMU (which returns cat of [fm, fm_skip] => 2*skip_ch), combine with upsampled in_ch
            self.dec_convs.append(nn.Sequential(
                nn.Conv3d(in_ch + 2*skip_ch, skip_ch, 3, padding=1, bias=False),
                nn.InstanceNorm3d(skip_ch, affine=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv3d(skip_ch, skip_ch, 3, padding=1, bias=False),
                nn.InstanceNorm3d(skip_ch, affine=True),
                nn.LeakyReLU(0.1, inplace=True),
            ))
            in_ch = skip_ch

        # heads
        self.main_head = nn.Conv3d(in_ch, out_channels, 1)
        self.deep_supervision = deep_supervision
        self.aux_heads = nn.ModuleList()
        if deep_supervision:
            # attach ds_heads from coarse->fine (skip last main head)
            for i in range(ds_heads):
                self.aux_heads.append(nn.Conv3d(dec_chs[i], out_channels, 1))

    def forward(self, x):
        # encoder
        f3d0 = self.enc3d_0(x)
        f2d0 = self.enc2d_0(x)
        f0 = torch.cat([f3d0, f2d0], dim=1)
        f0 = self.fuse0(f0)
        skips3d = [f3d0]; skips2d = [f2d0]
        f = self.down0(f0)
        for i in range(len(self.enc3d)):
            f3d = self.enc3d[i](f)
            f2d = self.enc2d[i](f)
            skips3d.append(f3d); skips2d.append(f2d)
            if i < len(self.enc3d)-1:
                f = self.downs[i](f3d)  # pass 3D-refined downwards (choice; symmetrical also ok)
            else:
                f = f3d

        z = self.bott(f)

        # decoder with FMU and deep supervision
        aux_logits = []
        in_feat = z
        for i, dec in enumerate(self.dec_convs):
            # pick the corresponding skip (deepest first)
            s3 = skips3d[-(i+1)]
            s2 = skips2d[-(i+1)]
            target_size = s3.shape[2:]  # (D,H,W) we want to align to this scale
            # upsample current decoder features exactly to the skip size
            up = F.interpolate(in_feat, size=target_size, mode="trilinear", align_corners=False)
            # FMU: also align 2D/3D skip features to the same target before abs-diff
            fm = self.fmus[i](s2, s3, target_size=target_size, skip2d=s2, skip3d=s3)

            # # upsample current features (trilinear, anisotropy-aware: (1,2,2))
            # up = F.interpolate(in_feat, scale_factor=(1,2,2), mode="trilinear", align_corners=False)
            # # select skip level (reverse order)
            # s3 = skips3d[-(i+1)]
            # s2 = skips2d[-(i+1)]
            # target_size = up.shape[2:]  # (D,H,W)
            # fm = self.fmus[i](s2, s3, target_size=target_size, skip2d=s2, skip3d=s3) # abs-diff fusion + skip fusion
            # fm = self.fmus[i](s2, s3, skip2d=s2, skip3d=s3)  
            x = torch.cat([up, fm], dim=1)
            in_feat = dec(x)
            # collect DS from coarse->fine
            if self.deep_supervision and i < len(self.aux_heads):
                aux_logits.append(self.aux_heads[i](in_feat))

        logits = self.main_head(in_feat)
        if self.deep_supervision:
            return {"main": logits, "aux": aux_logits}  # aux: coarse->fine
        return logits


if __name__ == "__main__":
    # pass

    net = MNet(in_channels=1, out_channels=2, base_ch=32, depth=4, deep_supervision=True, ds_heads=3)
    x = torch.randn(2,1,16,128,128)
    out = net(x)
    main = out["main"]
    print("main logits:", main.shape)  # expect [2, 2, 16, 128, 128]
    for i, a in enumerate(out["aux"]):
        print(f"aux[{i}] logits:", a.shape)  # descending resolutions that never exceed main