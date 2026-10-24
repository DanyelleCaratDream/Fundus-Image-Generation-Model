#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 RETFound-CFP（MAE ViT-L/16）提取冻结特征，供检索/重心条件使用

输出：CFM/data/retfound_cfp/feats.npy + names.json（顺序与图像列表一致）

用法
    python tools/retfound_features.py --src ../../../fundus/combined_sq256
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

CFM = Path(__file__).resolve().parents[1]
CKPT = CFM / "models/retfound_cfp/RETFound_mae_natureCFP.pth"
SEVERE_DIR = Path(r"D:/AI_Model_Project_for_Fundus_Color_Images/fundus/HRSG_4_758_sq256/train/severe")


def build_model(device):
    import timm
    model = timm.create_model("vit_large_patch16_224", pretrained=False,
                              num_classes=0, global_pool="token")
    ck = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    sd = ck["model"]
    # 只取编码器权重；MAE 的 decoder_* 与 head 不加载
    enc = {k: v for k, v in sd.items() if not k.startswith("decoder_") and k != "mask_token"
           and not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(enc, strict=False)
    print(f"[retfound] loaded encoder: missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        print("  missing sample:", missing[:5])
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=str(CFM / "data/retfound_cfp"))
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    tf = transforms.Compose([
        transforms.Resize((a.size, a.size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    files = sorted([p for p in Path(a.src).iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    print(f"[data] {len(files)} images")
    feats = []
    with torch.no_grad():
        for i in range(0, len(files), a.batch):
            batch = torch.stack([tf(Image.open(f).convert("RGB")) for f in files[i:i + a.batch]])
            feats.append(model(batch.to(device)).float().cpu())
            if (i // a.batch) % 3 == 0:
                print(f"  {i + len(batch)}/{len(files)}", flush=True)
    feats = torch.cat(feats).numpy()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "feats.npy", feats)
    (out / "names.json").write_text(json.dumps([f.name for f in files], ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    print(f"[done] {feats.shape} -> {out}")

    #  sanity：severe vs normal 的特征可分性
    sev = {f.stem for f in SEVERE_DIR.iterdir() if "_aug_1" not in f.name}
    labels = np.array([1 if f.stem in sev else 0 for f in files])
    fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    sim = fn @ fn.T
    same = sim[np.ix_(labels == 1, labels == 1)]
    cross = sim[np.ix_(labels == 1, labels == 0)]
    print(f"[sanity] severe={labels.sum()} normal={(labels==0).sum()} "
          f"within-severe cos={same.mean():.3f} cross={cross.mean():.3f}")


if __name__ == "__main__":
    main()
