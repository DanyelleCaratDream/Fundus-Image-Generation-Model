"""
Fundus-Diffusion: 从眼底彩照提取血管骨架图（条件扩散的条件输入）
============================================================
方法：
  1. 提取绿色通道（血管对比度最高）
  2. CLAHE 增强局部对比度
  3. Top-hat 形态学变换增强暗血管
  4. 自适应阈值分割
  5. 形态学去噪 + 骨架化

输出：与输入图同尺寸的单通道 PNG（白=血管，黑=背景）
"""

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm


def extract_vessel_skeleton(image_path, output_path, img_size=128):
    """
    从单张眼底图中提取血管骨架。

    Args:
        image_path: 输入眼底图路径
        output_path: 输出骨架图路径
        img_size: 输出的目标尺寸
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [跳过] 无法读取: {image_path}")
        return False

    # 1. 缩放到目标尺寸
    img = cv2.resize(img, (img_size, img_size))

    # 2. 提取绿色通道（眼底血管对比度最高的通道）
    green = img[:, :, 1]

    # 3. CLAHE 增强局部对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    # 4. 中值滤波去噪
    blurred = cv2.medianBlur(enhanced, 5)

    # 5. Top-hat 变换提取暗血管结构
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)

    # 6. 自适应阈值分割
    binary = cv2.adaptiveThreshold(
        tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 2
    )

    # 7. 形态学去噪（去掉孤立小点）
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean, iterations=1)

    # 8. 形态学闭合（填补断裂的血管）
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # 9. 骨架化（细化到单像素宽）
    skeleton = cv2.ximgproc.thinning(closed)

    # cv2.ximgproc 可能不可用，回退到形态学骨架
    if skeleton is None or skeleton.sum() < 100:
        # 手动骨架化：重复侵蚀直到单像素宽
        skeleton = _morphological_skeleton(closed)

    # 10. 保存
    cv2.imwrite(output_path, skeleton)
    return True


def extract_vessel_mask(image_path, output_path, img_size=128):
    """
    提取血管掩膜（非骨架版本，更鲁棒）。
    当骨架化效果不好时使用此版本。
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    img = cv2.resize(img, (img_size, img_size))
    green = img[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    blurred = cv2.medianBlur(enhanced, 5)

    # Top-hat 提取暗结构
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)

    # 自适应阈值
    binary = cv2.adaptiveThreshold(
        tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 2
    )

    # 去噪 + 闭合
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean, iterations=1)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    cv2.imwrite(output_path, mask)
    return True


def _morphological_skeleton(binary_img):
    """形态学骨架化（不需要 ximgproc）"""
    skeleton = np.zeros_like(binary_img)
    img = binary_img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, kernel)
        if eroded.sum() == 0:
            break
        dilated = cv2.dilate(eroded, kernel)
        skeleton += cv2.subtract(img, dilated)
        img = eroded

    return skeleton


def main():
    parser = argparse.ArgumentParser(
        description="Fundus-Diffusion: 提取眼底血管骨架作为条件扩散的条件输入",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="原始眼底图路径")
    parser.add_argument("--output_dir", type=str, default="./conditions", help="骨架图输出目录")
    parser.add_argument("--img_size", type=int, default=128, help="输出尺寸")
    parser.add_argument("--mode", type=str, default="mask",
                        choices=["skeleton", "mask"],
                        help="skeleton=单像素骨架, mask=血管掩膜(更鲁棒)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 收集图片
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    image_files = sorted([
        f for f in os.listdir(args.dataset_path)
        if f.lower().endswith(exts)
    ])

    print(f"找到 {len(image_files)} 张图片")
    print(f"模式: {args.mode}")
    print(f"输出: {args.output_dir}")
    print()

    success = 0
    for fname in tqdm(image_files, desc="提取血管"):
        in_path = os.path.join(args.dataset_path, fname)
        out_name = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(args.output_dir, out_name)

        if args.mode == "skeleton":
            ok = extract_vessel_skeleton(in_path, out_path, args.img_size)
        else:
            ok = extract_vessel_mask(in_path, out_path, args.img_size)

        if ok:
            success += 1

    print(f"\n完成: {success}/{len(image_files)} 张成功")
    return 0


if __name__ == "__main__":
    exit(main())
