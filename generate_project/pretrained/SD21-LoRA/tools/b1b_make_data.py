#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""重建 B1b 训练数据：老师 severe train 图 → 512×512 LANCZOS（b1b_data/severe512）

源 = fundus/HRSG_4_758_square/train/severe（老师数据只读纪律：本脚本只写副本目录），
跳过 _aug_1 增广文件（保持 72 张唯一底图）。
b1b_data/ 不入 git（.gitignore），组员克隆仓库后先跑本脚本，再训 B1b。

用法
    F:/python/python.exe b1b_make_data.py
"""
from pathlib import Path

from PIL import Image

SRC = Path(r"D:\AI_Model_Project_for_Fundus_Color_Images\fundus\HRSG_4_758_square\train\severe")
OUT = Path(__file__).resolve().parents[1] / "data" / "b1b_data" / "severe512"


def main():
    files = sorted(f for f in SRC.glob("*.jpg") if "_aug_1" not in f.name)
    assert files, f"no source images in {SRC}"
    OUT.mkdir(parents=True, exist_ok=True)
    for f in files:
        img = Image.open(f).convert("RGB").resize((512, 512), Image.LANCZOS)
        img.save(OUT / f.name, quality=95)
    print(f"[done] {len(files)} -> {OUT}")


if __name__ == "__main__":
    main()
