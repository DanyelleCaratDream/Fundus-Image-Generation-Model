# -*- coding: utf-8 -*-
"""
旋转增强检查工具：生成图片旋转 45/90/135/180 度的四个规格，
并合成一张 2×2 对比图，供人工确认旋转后眼底圆是否仍然完整。

用法：
  1. 把本脚本和你想要检查的图片放在同一个文件夹里
  2. 在终端运行:  python rotate_augment_check.py
  3. 输入图片文件名（可空格隔开多张，直接回车结束）

输出：
  生成在脚本所在目录的上一层目录 rotate_augment_check/ 文件夹下
  （若该文件夹已存在则直接复用，不重复创建）
  每张原图输出一张 2×2 对比图: {原图名}_rot2x2.{原格式}
  布局: 左上=45°  右上=90°  左下=135°  右下=180°
"""
import os
from PIL import Image

# 脚本所在目录（与图片同目录），以脚本自身位置为基准，拷到哪都能用
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 输出目录：脚本所在目录的上一层目录，命名由本脚本决定
OUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "rotate_augment_check")

ANGLES = [45, 90, 135, 180]

# 支持常见图片格式
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"}


def make_grid(rotated):
    """把 4 张旋转图合成一张 2×2 对比图。

    每格取 4 张中的最大边长，小图居中粘贴、四周黑底补齐，
    只加黑边不缩放，保证旋转内容原分辨率无损。
    """
    cell = max(max(im.width, im.height) for im in rotated)
    grid = Image.new("RGB", (cell * 2, cell * 2), (0, 0, 0))
    for i, im in enumerate(rotated):
        x = (i % 2) * cell + (cell - im.width) // 2
        y = (i // 2) * cell + (cell - im.height) // 2
        grid.paste(im, (x, y))
    return grid


def process_one(fname):
    """处理单张图片：4 个角度旋转并合成 2×2，成功返回1否则0。"""
    name, ext = os.path.splitext(fname)
    src_path = os.path.join(BASE_DIR, fname)

    if os.path.splitext(fname.lower())[1] not in EXTS:
        print(f"  [跳过] 不支持的格式: {fname}")
        return 0
    if not os.path.isfile(src_path):
        print(f"  [跳过] 找不到文件: {fname}")
        return 0

    try:
        with Image.open(src_path) as img:
            # 转换为RGB（去除透明通道等，保持兼容性）
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            rotated = []
            for ang in ANGLES:
                # expand=True 保留完整旋转内容(45/135度画布变大)，才能看出圆是否被裁出画面
                # 眼底图背景为黑色，黑底填充无痕，不干扰判断
                rot = img.rotate(ang, expand=True,
                                 resample=Image.BICUBIC, fillcolor=(0, 0, 0))
                rotated.append(rot)

            grid = make_grid(rotated)
            out_path = os.path.join(OUT_DIR, f"{name}_rot2x2{ext}")
            grid.save(out_path, quality=95)
        print(f"  [完成] {fname} -> {os.path.basename(out_path)}")
        return 1
    except Exception as e:
        print(f"  [失败] {fname}: {e}")
        return 0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)  # 已存在则复用，不重复创建
    print(f"输出目录: {OUT_DIR}\n")
    total = 0

    while True:
        line = input("请输入图片文件名（可空格隔开多张，直接回车退出）: ").strip()
        if not line:
            break
        for fname in line.split():
            total += process_one(fname)
        print()

    print(f"全部完成，共处理 {total} 张图片。\n输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
