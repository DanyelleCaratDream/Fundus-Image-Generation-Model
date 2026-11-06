#!/usr/bin/env python3
"""RETOUCH (.mhd/.raw) -> .npy 转换脚本

将 RETOUCH 官方下载的 MetaImage 格式数据转换为 TTA-Flow 官方仓库所期望的
NumPy 布局：

    <out_root>/
    ├── <device>/            # spectralis / cirrus / topcon (小写)
    │   ├── 001.npy          # 3D 灰度体积 (D, H, W), float32, 值域 [0, 1]
    │   └── ...
    └── annotations/
        └── <device>/
            ├── 001.npy      # 3D 多类掩膜 (D, H, W), uint8, 0=背景 1=IRF 2=SRF 3=PED
            └── ...

设计要点
--------
1. **布局自适应**：官方 RETOUCH 在不同年份/镜像中的目录层级略有差异（可能直接在设备目录下，
   也可能每个体积一个子目录），因此本脚本用 rglob 发现 `oct.mhd` / `reference.mhd` 配对，
   不硬编码层级。
2. **轴序**：SimpleITK 返回 (z, y, x)。RETOUCH 的 B-scan 位于 z 轴，因此 (z, y, x) 恰好
   对应仓库期望的 (D, H, W)，无需转置；但本脚本仍打印形状供人工核对。
3. **标签编码**：RETOUCH 官方 reference 存在 4 类（0/1/2/3）与 6 类两种编码版本。脚本会
   打印实际出现的唯一值分布，并在 6 类编码时按官方映射归并为 1/2/3，避免静默错标。
4. **归一化**：落盘值域为 **[0, 1]**。依据是仓库 `preferences/transforms/transforms_fm.yaml`
   中的 `Normalize(mean=0.5, std=0.5, max_pixel_value=1.0)` —— 该变换计算
   `(x - 0.5) / 0.5 = 2x - 1`，只有输入为 [0, 1] 时才得到模型所需的 [-1, 1]；
   若输入误用 [-1, 1]，输出会被压成 [-3, 1]，与生成模型的零均值高斯先验不匹配。
   此处按体积做 min-max 归一化后落盘，
   保证与仓库 `transforms_fm.yaml` 的预期值域一致。

用法
----
    python retouch_to_npy.py --data-root /path/to/RETOUCH --out-root /path/to/retouch_npy
    python retouch_to_npy.py --data-root ... --out-root ... --dry-run   # 只检查不写盘
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover
    sys.exit("缺少 SimpleITK，请先执行: pip install SimpleITK")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("retouch2npy")

# 官方 6 类编码 -> 3 类语义 (IRF/SRF/PED) 的归并映射。
# RETOUCH 训练集标注中，视网膜内液/视网膜下液/色素上皮脱离分别有多套子类编号。
SIX_CLASS_TO_THREE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2}

# 设备目录名归一化：把各种写法收敛到仓库 CSV 使用的小写名
DEVICE_ALIASES = {
    "spectralis": "spectralis",
    "heidelberg": "spectralis",
    "cirrus": "cirrus",
    "zeiss": "cirrus",
    "topcon": "topcon",
    "triton": "topcon",
}


@dataclass
class VolumePair:
    """一个体积的图像/标注配对及推断出的设备名。"""

    image: Path
    label: Path | None
    device: str
    stem: str

    def __str__(self) -> str:  # pragma: no cover - 仅用于日志
        return f"{self.device}/{self.stem}"


@dataclass
class Stats:
    """转换过程中的统计信息，用于最终的一致性核对。"""

    found: Counter = field(default_factory=Counter)
    converted: Counter = field(default_factory=Counter)
    label_histogram: Counter = field(default_factory=Counter)
    shapes: list = field(default_factory=list)
    missing_labels: list = field(default_factory=list)
    failed: list = field(default_factory=list)


def pad_stem(stem: str) -> str:
    """把纯数字的体积名补零到 3 位，与官方仓库的 `001.npy` 命名对齐。

    RETOUCH 官方目录名多为裸数字（1, 2, ...）。补零不影响功能（生成的 CSV 指向
    实际写出的文件名即可），但能让输出与仓库 `data/dataset_configs/*.csv` 里
    `spectralis/001.npy` 的写法一致，便于人工比对。
    """
    return stem.zfill(3) if stem.isdigit() else stem


def volume_sort_key(pair: "VolumePair") -> tuple:
    """按 (设备, 是否数字, 数值/名) 排序，避免字符串排序把 10 排到 9 前面。"""
    name = pair.stem
    return (pair.device, 0, int(name)) if name.isdigit() else (pair.device, 1, name)



def infer_device(path: Path) -> str:
    """从文件路径中推断设备名。找不到时返回 'unknown'。"""
    for part in reversed(path.parts):
        key = part.strip().lower()
        if key in DEVICE_ALIASES:
            return DEVICE_ALIASES[key]
        for alias, canonical in DEVICE_ALIASES.items():
            if alias in key:
                return canonical
    return "unknown"


def discover(data_root: Path) -> list[VolumePair]:
    """发现所有 (图像, 标注) 配对。

    以 `oct.mhd` 作为图像锚点；同目录下的 `reference.mhd` 作为标注。
    若不存在 `oct.mhd`（部分镜像用别的命名），退化为：把所有 .mhd 按目录分组，
    目录内文件名含 'reference' 的视为标注、其余视为图像。
    """
    pairs: list[VolumePair] = []

    # 主路径：标准命名
    for img in sorted(data_root.rglob("oct.mhd")):
        ref = img.parent / "reference.mhd"
        pairs.append(
            VolumePair(
                image=img,
                label=ref if ref.exists() else None,
                device=infer_device(img),
                stem=pad_stem(img.parent.name),
            )
        )


    # 备用路径：非标准命名，按目录分组
    if not pairs:
        log.warning("未找到标准的 oct.mhd，改用按目录分组的备用发现逻辑")
        by_dir: dict[Path, list[Path]] = {}
        for mhd in sorted(data_root.rglob("*.mhd")):
            by_dir.setdefault(mhd.parent, []).append(mhd)
        for directory, files in sorted(by_dir.items()):
            refs = [f for f in files if "reference" in f.name.lower() or "label" in f.name.lower()]
            imgs = [f for f in files if f not in refs]
            for img in imgs:
                pairs.append(
                    VolumePair(
                        image=img,
                        label=refs[0] if refs else None,
                        device=infer_device(img),
                        stem=pad_stem(directory.name),
                    )
                )

    # 统一排序：设备名 -> 体积数值序，保证 CSV 行序稳定可复现
    pairs.sort(key=volume_sort_key)
    if pairs:
        log.info("发现 %d 个体积（按设备: %s）", len(pairs), dict(Counter(p.device for p in pairs)))

    # 统一去重（同一 stem 在同一设备下只保留一次）
    seen: set[tuple[str, str]] = set()
    unique: list[VolumePair] = []
    for p in pairs:
        key = (p.device, p.stem)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def read_mhd(path: Path) -> np.ndarray:
    """读取 MetaImage 文件为 NumPy 数组（SimpleITK 返回 (z, y, x)）。"""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def normalize_to_unit_range(volume: np.ndarray) -> np.ndarray:
    """将体积 min-max 归一化到 [0, 1]。

    值域选择依据见文件头说明：仓库的 Normalize 变换会把 [0, 1] 映射为 [-1, 1]，
    即 [0, 1] 才是磁盘上的正确约定。归一化在**整卷**上做（不逐 B-scan），
    以保留体积内部的相对强度一致性。
    """
    vol = volume.astype(np.float32)
    lo, hi = float(vol.min()), float(vol.max())
    if hi - lo < 1e-8:
        log.warning("体积动态范围过小 (min=%.4f, max=%.4f)，返回全零", lo, hi)
        return np.zeros_like(vol)
    return (vol - lo) / (hi - lo)


def remap_labels(mask: np.ndarray, stats: Stats) -> np.ndarray:
    """把标注归并到 0/1/2/3 四类，并记录原始唯一值分布。"""
    uniq, counts = np.unique(mask, return_counts=True)
    for u, c in zip(uniq.tolist(), counts.tolist()):
        stats.label_histogram[int(u)] += int(c)

    present = set(int(u) for u in uniq)
    out = mask.astype(np.uint8)
    if present - {0, 1, 2, 3} and present <= set(SIX_CLASS_TO_THREE):
        log.info("检测到 6 类标注编码，按官方映射归并到 0/1/2/3")
        remapped = np.zeros_like(out)
        for src, dst in SIX_CLASS_TO_THREE.items():
            if src:
                remapped[out == src] = dst
        out = remapped
    return out


def convert(
    pairs: list[VolumePair],
    out_root: Path,
    dry_run: bool = False,
    overwrite: bool = False,
) -> Stats:
    """执行转换并返回统计信息。"""
    stats = Stats()

    for i, pair in enumerate(pairs, start=1):
        stats.found[pair.device] += 1
        try:
            image = read_mhd(pair.image)
            stats.shapes.append(tuple(image.shape))
        except Exception as exc:  # noqa: BLE001 - 需要记录任意读取失败
            log.error("[%d/%d] 读取失败 %s: %s", i, len(pairs), pair, exc)
            stats.failed.append(str(pair))
            continue

        label: np.ndarray | None = None
        if pair.label is not None:
            try:
                label = remap_labels(read_mhd(pair.label), stats)
            except Exception as exc:  # noqa: BLE001
                log.error("[%d/%d] 标注读取失败 %s: %s", i, len(pairs), pair, exc)
                stats.failed.append(str(pair))
                continue
        else:
            stats.missing_labels.append(str(pair))

        if label is not None and label.shape != image.shape:
            log.error(
                "[%d/%d] 形状不匹配 %s: image=%s label=%s",
                i, len(pairs), pair, image.shape, label.shape,
            )
            stats.failed.append(str(pair))
            continue

        if dry_run:
            stats.converted[pair.device] += 1
            continue

        vol_dir = out_root / pair.device
        ann_dir = out_root / "annotations" / pair.device
        vol_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)

        vol_path = vol_dir / f"{pair.stem}.npy"
        if vol_path.exists() and not overwrite:
            log.debug("已存在，跳过: %s", vol_path)
            stats.converted[pair.device] += 1
            continue

        np.save(vol_path, normalize_to_unit_range(image))
        if label is not None:
            np.save(ann_dir / f"{pair.stem}.npy", label)

        stats.converted[pair.device] += 1
        if i % 10 == 0 or i == len(pairs):
            log.info("进度 %d/%d，已转换 %d", i, len(pairs), sum(stats.converted.values()))

    return stats


def verify(out_root: Path, stats: Stats) -> bool:
    """转换后核对：数量、形状一致性、值域、标签类别。返回是否通过。"""
    log.info("=" * 68)
    log.info("转换核对")
    log.info("=" * 68)

    ok = True

    # 1. 数量核对（以论文/仓库划分作为参照）
    expected = {"spectralis": 24, "cirrus": 24, "topcon": 22}
    for device, exp in expected.items():
        got = len(list((out_root / device).glob("*.npy")))
        flag = "✅" if got == exp else "⚠️"
        if got != exp:
            ok = False
        log.info("%s %-12s 体积数 = %d（论文/仓库划分期望 %d）", flag, device, got, exp)

    # 2. 形状一致性
    if stats.shapes:
        shape_set = set(stats.shapes)
        log.info("读取到的体积形状种类: %s", shape_set if len(shape_set) < 8 else f"{len(shape_set)} 种")

    # 3. 值域与标签类别抽查
    sample = sorted((out_root / "spectralis").glob("*.npy"))[:1]
    if sample:
        vol = np.load(sample[0])
        log.info(
            "抽查 %s: shape=%s dtype=%s range=[%.3f, %.3f]",
            sample[0].name, vol.shape, vol.dtype, vol.min(), vol.max(),
        )
        if not (-1.001 <= float(vol.min()) and float(vol.max()) <= 1.001):
            log.error("值域超出 [0, 1]，归一化有误")
            ok = False

    ann_sample = sorted((out_root / "annotations" / "spectralis").glob("*.npy"))[:1]
    if ann_sample:
        m = np.load(ann_sample[0])
        present = sorted(int(v) for v in np.unique(m))
        log.info("抽查标注 %s: shape=%s 类别=%s", ann_sample[0].name, m.shape, present)
        unexpected = set(present) - {0, 1, 2, 3}
        if unexpected:
            log.error("出现预期外的标签类别: %s（RETOUCH 应为 0/1/2/3）", unexpected)
            ok = False

    # 4. 标签体素总占比（三设备合并），用于判断标签编码是否合理
    if stats.label_histogram:
        total = sum(stats.label_histogram.values())
        log.info("标注体素占比（全部设备合并）:")
        for label in sorted(stats.label_histogram):
            frac = stats.label_histogram[label] / total
            name = {0: "background", 1: "IRF", 2: "SRF", 3: "PED"}.get(label, "unknown")
            log.info("  %-10s (值=%d): %6.4f%%", name, label, frac * 100)
        irf = stats.label_histogram.get(1, 0) / total
        if irf == 0:
            log.error("IRF 体素占比为 0，标签映射很可能有误")
            ok = False

    if stats.missing_labels:
        log.warning("有 %d 个体积缺少标注（仅影响下游评估，不影响生成模型训练）", len(stats.missing_labels))

    if stats.failed:
        log.error("失败 %d 个: %s", len(stats.failed), stats.failed[:10])
        ok = False

    return ok


def write_manifest(out_root: Path, stats: Stats) -> None:
    """写出转换清单，供复现报告与追溯使用。"""
    manifest = {
        "found_per_device": dict(stats.found),
        "converted_per_device": dict(stats.converted),
        "label_voxel_histogram": {str(k): v for k, v in sorted(stats.label_histogram.items())},
        "volume_shapes": sorted({str(s) for s in stats.shapes}),
        "volumes_without_annotation": stats.missing_labels,
        "failed": stats.failed,
    }
    path = out_root / "conversion_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("转换清单已写出: %s", path)


def main() -> int:
    parser = argparse.ArgumentParser(description="RETOUCH .mhd/.raw -> .npy 转换")
    parser.add_argument("--data-root", required=True, type=Path, help="RETOUCH 解压后的根目录")
    parser.add_argument("--out-root", required=True, type=Path, help="输出根目录")
    parser.add_argument("--dry-run", action="store_true", help="只发现与检查，不写盘")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 .npy")
    args = parser.parse_args()

    if not args.data_root.exists():
        log.error("数据根目录不存在: %s", args.data_root)
        return 2

    log.info("扫描 %s ...", args.data_root)
    pairs = discover(args.data_root)
    log.info("发现 %d 个待转换体积", len(pairs))
    if not pairs:
        log.error("未发现任何体积，请检查 --data-root 是否指向 RETOUCH 解压目录")
        return 3

    per_device = Counter(p.device for p in pairs)
    log.info("按设备分布: %s", dict(per_device))
    if per_device.get("unknown"):
        log.warning("有 %d 个体积无法推断设备（目录名不含 spectralis/cirrus/topcon），请核对",
                    per_device["unknown"])

    args.out_root.mkdir(parents=True, exist_ok=True)
    stats = convert(pairs, args.out_root, dry_run=args.dry_run, overwrite=args.overwrite)

    if args.dry_run:
        log.info("dry-run 完成，未写盘")
        return 0

    ok = verify(args.out_root, stats)
    write_manifest(args.out_root, stats)

    log.info("=" * 68)
    log.info("结果: %s", "✅ 通过" if ok else "❌ 存在问题，请按上方日志排查后再进入下一步")
    log.info("下一步: 运行 create_dataset_configs.py 生成仓库所需的 CSV 划分文件")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
