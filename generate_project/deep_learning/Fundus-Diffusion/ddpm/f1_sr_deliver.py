#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""F1 交付管线：D3 @128 → Real-ESRGAN x4 → 512 → LANCZOS 256

为什么 x4 再降采样：超分后降采样（supersample）比直接 x2 更锐，且 x4plus 权重
是 Real-ESRGAN 官方主模型（23 个 RRDB，64 特征），质量最稳。

对照组（同一批 8 张）：真实 severe / D3 原图双三次放大 / D3 原图+SR / D3+SDEdit 双三次 / D3+SDEdit+SR
产出：judgment_20261020/D3_sr/{orig_256,t00200_256}/ + f1_sr_compare.png
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
J = ROOT / "judgment_20261020"
S3 = ROOT / "sdedit_D3_ema"
WEIGHTS = Path(r"D:\tmp\sr_weights\RealESRGAN_x4.pth")
OUT = J / "D3_sr"


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

    def forward(self, x):
        a = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        b = F.leaky_relu(self.conv2(torch.cat((x, a), 1)), 0.2, inplace=True)
        c = F.leaky_relu(self.conv3(torch.cat((x, a, b), 1)), 0.2, inplace=True)
        d = F.leaky_relu(self.conv4(torch.cat((x, a, b, c), 1)), 0.2, inplace=True)
        e = self.conv5(torch.cat((x, a, b, c, d), 1))
        return e * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRGAN x4plus：23 RRDB，64 特征"""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = F.interpolate(feat, scale_factor=2, mode="nearest")
        feat = F.leaky_relu(self.conv_up1(feat), 0.2, inplace=True)
        feat = F.interpolate(feat, scale_factor=2, mode="nearest")
        feat = F.leaky_relu(self.conv_up2(feat), 0.2, inplace=True)
        feat = F.leaky_relu(self.conv_hr(feat), 0.2, inplace=True)
        return self.conv_last(feat)


@torch.no_grad()
def upscale(model, img: Image.Image) -> Image.Image:
    x = torch.from_numpy(
        __import__("numpy").asarray(img.convert("RGB"), dtype="float32") / 255.0
    ).permute(2, 0, 1).unsqueeze(0).cuda()
    y = model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((y * 255.0).round().astype("uint8"))


def main():
    model = RRDBNet().cuda().eval()
    sd = torch.load(str(WEIGHTS), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[sr] loaded {WEIGHTS.name}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("orig_256", "t00200_256", "orig_bicubic", "t00200_bicubic"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    for src_name, out_tag in (("orig", "orig"), ("t00200", "t00200")):
        files = sorted((S3 / src_name).glob("sample_*.png"))
        for f in files:
            im = Image.open(f).convert("RGB")
            bic = im.resize((256, 256), Image.BICUBIC)
            bic.save(OUT / f"{out_tag}_bicubic" / f.name)
            sr = upscale(model, im).resize((256, 256), Image.LANCZOS)
            sr.save(OUT / f"{out_tag}_256" / f.name)
            print(f"[ok] {out_tag}/{f.name}", flush=True)

    # 五行对比：真实 / D3 双三次 / D3+SR / SDEdit 双三次 / SDEdit+SR
    real = sorted((ROOT / "b1b_data" / "severe512").glob("*.jpg"))[:8]
    rows = [
        (real, "real severe (512)"),
        (sorted((OUT / "orig_bicubic").glob("*.png")), "D3 raw 128 -> bicubic 256"),
        (sorted((OUT / "orig_256").glob("*.png")), "D3 raw 128 -> Real-ESRGAN x4 -> 256"),
        (sorted((OUT / "t00200_bicubic").glob("*.png")), "D3 + SDEdit t0=200 -> bicubic 256"),
        (sorted((OUT / "t00200_256").glob("*.png")), "D3 + SDEdit t0=200 -> Real-ESRGAN x4 -> 256"),
    ]
    S, PAD, CAP, N = 256, 6, 24, 8
    canvas = Image.new("RGB", (N * S + (N + 1) * PAD, len(rows) * (S + CAP + PAD) + PAD), "white")
    dr = ImageDraw.Draw(canvas)
    y = PAD
    for files, label in rows:
        dr.text((PAD, y), label, fill="black"); y += CAP
        for i, f in enumerate(files[:N]):
            canvas.paste(Image.open(f).convert("RGB").resize((S, S), Image.BICUBIC),
                         (PAD + i * (S + PAD), y))
        y += S + PAD
    canvas.save(J / "f1_sr_compare.png")
    print(f"[done] {J / 'f1_sr_compare.png'}", flush=True)


if __name__ == "__main__":
    main()
