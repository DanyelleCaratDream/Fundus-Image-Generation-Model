#!/usr/bin/env python3
"""下游分割任务的共享组件：模型构建、数据集、Dice 指标。

论文设定（第 3 节 Experimental Setup）
--------------------------------------
* 病灶分割（IRF/SRF/PED）：UNet++ + ResNet-18 编码器，**仅在 RETOUCH Spectralis 子集上训练**，
  用于评测各重建方法的质量；论文称其为 "downstream model"。
* 评估范围：仅在被标注的 B-scan 上统计 Dice（DSC）。

本模块同时被 `train_seg.py`（训练/交叉验证）、`eval_dice.py`（评测重建结果）与
`baselines/eval_tta_seg.py`（TENT/CoTTA 基线）复用，以保证三者的数据处理、归一化、
类别顺序完全一致——这是让 DSC 可比的前提。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# 类别顺序固定在全局，训练与评测共用；改动此处会同时影响两侧，避免错位
CLASS_NAMES = ("IRF", "SRF", "PED")
NUM_CLASSES = len(CLASS_NAMES)
# 体积内用于区分前景的通道偏移（0 为背景）
#   mask 原始值 1/2/3 -> 通道 0/1/2
LABEL_TO_CHANNEL = {1: 0, 2: 1, 3: 2}

# B-scan 的网络输入尺寸。论文为 224×224；仓库默认配置为 128×128（见复现计划 R1）。
DEFAULT_INPUT_SIZE = 224

# 统一值域约定（重要，已按仓库源码核实）
# ---------------------------------------------------------------------------
# 磁盘上的 `.npy` 体积与推理输出 `pred_*.npy` **都是 [0, 1]**：
#   * 训练侧：`transforms_fm.yaml` 的 `Normalize(mean=0.5, std=0.5, max_pixel_value=1.0)`
#     计算 (x - 0.5) / 0.5，只有输入为 [0, 1] 时才得到模型所需的 [-1, 1]；
#   * 推理侧：`src/test.py` 写出前做 `out_vol.clip(-1, 1) / 2 + 0.5`，把模型的 [-1, 1]
#     映回 [0, 1]，与磁盘约定一致。
# 因此下游分割模型统一以 **[0, 1]** 为输入约定，训练与评测共用本文件的 convert_range()。
# 若将来改用其它约定（例如直接喂 [-1, 1]），训练与评测的 --image-range 必须同时改，
# 否则 Dice 会静默劣化且不报错。
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_RANGE = "0-1"
_NPY_RANGE = "0-1"
VALID_RANGES = ("0-1", "-1-1")


def convert_range(img: np.ndarray, src: str = _NPY_RANGE, dst: str = DEFAULT_IMAGE_RANGE) -> np.ndarray:
    """在 [-1, 1] 与 [0, 1] 之间转换图像值域。"""
    if src not in VALID_RANGES or dst not in VALID_RANGES:
        raise ValueError(f"未知值域: src={src!r}, dst={dst!r}")
    if src == dst:
        return img
    if src == "-1-1" and dst == "0-1":
        return (img + 1.0) / 2.0
    if src == "0-1" and dst == "-1-1":
        return img * 2.0 - 1.0
    raise ValueError(f"不支持的值域转换: {src} -> {dst}")


def build_model(encoder_weights: str | None = "imagenet", input_size: int = DEFAULT_INPUT_SIZE):
    """构建 UNet++ / ResNet-18 分割模型。

    Args:
        encoder_weights: 传入 None 可避免联网下载预训练权重（离线环境用）。
        input_size: 仅用于日志提示，模型本身为全卷积结构。

    Returns:
        torch.nn.Module: 输出 NUM_CLASSES 个通道的 logits。
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "缺少 segmentation-models-pytorch，请执行: pip install segmentation-models-pytorch"
        ) from exc

    model = smp.UnetPlusPlus(
        encoder_name="resnet18",
        encoder_weights=encoder_weights,
        in_channels=1,
        classes=NUM_CLASSES,
        activation=None,  # 输出 logits，配合 BCEWithLogits 使用
    )
    log.info("已构建 UNet++ / ResNet-18（encoder_weights=%s, input_size=%d）",
             encoder_weights, input_size)
    return model


def mask_to_channels(mask: np.ndarray) -> np.ndarray:
    """把 (H, W) 的多类掩膜转成 (NUM_CLASSES, H, W) 的 0/1 多标签编码。"""
    out = np.zeros((NUM_CLASSES,) + mask.shape, dtype=np.float32)
    for label, ch in LABEL_TO_CHANNEL.items():
        out[ch] = (mask == label).astype(np.float32)
    return out


@dataclass
class SegSample:
    """一个 B-scan 样本及其来源信息，便于评测时按体积聚合。"""

    image: np.ndarray   # (1, H, W) float32, 值域由 Dataset.image_range 决定
    target: np.ndarray  # (NUM_CLASSES, H, W) float32, 0/1
    volume: str
    slice_index: int


class RetouchSegDataset(Dataset):
    """RETOUCH 体积 -> 2D B-scan 切片数据集。

    Args:
        csv_path: 两列 CSV（volume, mask），与官方仓库 `data/dataset_configs/*.csv` 同构。
        input_size: 缩放到的正方形边长。
        annotated_only: 只保留"被标注"的切片。优先使用 `annotated_slices` 显式清单；
            未提供清单时退化为"跳过掩膜全零的切片"，与 eval_dice.py 的默认口径一致
            （RETOUCH 中"未标注"与"已标注但无病灶"都表现为全零掩膜，无法区分）。
        annotated_slices: {volume 文件名: set(切片下标)}，可由 RETOUCH 官方的切片清单构建。
        max_volumes: 仅用于 pilot 冒烟测试，限制体积数量。
        image_range: 输出图像的值域，须与下游模型训练时一致。
    """

    def __init__(
        self,
        csv_path: Path,
        input_size: int = DEFAULT_INPUT_SIZE,
        annotated_only: bool = True,
        annotated_slices: dict[str, set[int]] | None = None,
        max_volumes: int | None = None,
        image_range: str = DEFAULT_IMAGE_RANGE,
    ) -> None:
        if image_range not in VALID_RANGES:
            raise ValueError(f"image_range 必须为 {VALID_RANGES} 之一，收到 {image_range!r}")
        self.input_size = input_size
        self.annotated_only = annotated_only
        self.annotated_slices = annotated_slices
        self.image_range = image_range

        df = pd.read_csv(csv_path, dtype=str)
        if max_volumes is not None:
            df = df.head(max_volumes)
        self.df = df.reset_index(drop=True)

        # 预扫描索引：(volume_idx, slice_idx)。
        # 缓存只保留"最近访问的一个体积"：RETOUCH 单个体积可达约 130 MB（Cirrus/Topcon
        # 128 层 × 512 × 496），全部缓存会吃光内存；而数据按体积顺序访问，单条缓存足够。
        self.index: list[tuple[int, int]] = []
        self._vol_i: int | None = None
        self._vol: np.ndarray | None = None
        self._msk_i: int | None = None
        self._msk: np.ndarray | None = None

        n_skipped = 0
        for vi in range(len(self.df)):
            mask_path = Path(self.df.loc[vi, "mask"])
            vol_path = Path(self.df.loc[vi, "volume"])
            vol_name = vol_path.name
            vol_key = f"{vol_path.parent.name}/{vol_name}"
            if mask_path.suffix != ".npy":
                raise ValueError(f"掩膜必须为 .npy 格式，收到 {mask_path.name}")
            n_slices = self._get_n_slices(vi)
            allowed = None
            if annotated_only and self.annotated_slices is not None:
                allowed = (self.annotated_slices.get(vol_key)
                           or self.annotated_slices.get(vol_name))
            # 无显式清单时，用掩膜是否非零来判断"是否已标注"
            mask = self._mask(vi) if (annotated_only and allowed is None) else None
            for si in range(n_slices):
                if allowed is not None:
                    if si not in allowed:
                        continue
                elif mask is not None and mask[si].sum() == 0:
                    n_skipped += 1
                    continue
                self.index.append((vi, si))
        if n_skipped:
            log.warning("跳过 %d 个掩膜全零的切片（退化口径；建议补充 RETOUCH 官方标注切片清单）",
                        n_skipped)

        log.info("数据集 %s: %d 个体积, %d 个 B-scan 样本 (input_size=%d, image_range=%s)",
                 Path(csv_path).name, len(self.df), len(self.index), input_size, image_range)

    def _get_n_slices(self, vi: int) -> int:
        return int(np.load(self.df.loc[vi, "volume"], mmap_mode="r").shape[0])

    def _volume(self, vi: int) -> np.ndarray:
        if vi != self._vol_i:
            self._vol = np.load(self.df.loc[vi, "volume"]).astype(np.float32)
            self._vol_i = vi
        return self._vol

    def _mask(self, vi: int) -> np.ndarray:
        if vi != self._msk_i:
            self._msk = np.load(self.df.loc[vi, "mask"]).astype(np.uint8)
            self._msk_i = vi
        return self._msk

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> SegSample:
        vi, si = self.index[idx]
        img = self._volume(vi)[si]                     # (H, W)
        msk = self._mask(vi)[si]                       # (H, W)

        img, msk = resize_pair(img, msk, self.input_size)

        # 磁盘 .npy 为 [0, 1]（见文件头的约定说明），转换到下游模型期望的值域后补通道维
        image = convert_range(img, _NPY_RANGE, self.image_range)[None, :, :].astype(np.float32)
        target = mask_to_channels(msk)
        return SegSample(image=image, target=target,
                         volume=Path(self.df.loc[vi, "volume"]).name, slice_index=si)


def resize_pair(img: np.ndarray, msk: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """把图像与掩膜同步缩放到 (size, size)。掩膜用最近邻以保持类别值不被插值污染。"""
    if img.shape == (size, size):
        return img, msk
    import cv2

    img_r = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    msk_r = cv2.resize(msk, (size, size), interpolation=cv2.INTER_NEAREST)
    return img_r, msk_r


def dice_per_class(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-6,
    is_logit: bool | None = None,
) -> np.ndarray:
    """逐类别计算 Dice 系数。

    与论文口径一致：对每个类别分别计算 2|X∩Y| / (|X|+|Y|)，空类（预测与真值均为空）
    记为 1.0（完全正确），这是分割评测中的通行约定。

    Args:
        pred:   (C, H, W) 或 (B, C, H, W)。未过 sigmoid 的 logits，或已二值化的 0/1。
        target: 两种约定都支持，且**同一批数据用两种约定算出的结果完全等价**：
                  * 通道格式 `(C,H,W)` / `(B,C,H,W)`：0/1 多标签，与 pred 同形；
                  * **整数类别图** `(H,W)` / `(B,H,W)`：0=背景，1..C 依次对应通道 0..C-1
                    （即 RETOUCH 掩膜的原生语义），内部按类别做 one-hot。
                第二种约定与 `LABEL_TO_CHANNEL`（1→IRF, 2→SRF, 3→PED）一致。
        threshold: 判定为前景的概率阈值（作用于 sigmoid 之后）。
        is_logit:  pred 是否为 logits。默认 None 表示自动判定（浮点且非纯 0/1 即视为 logits）。
                   **建议调用方显式传入**，以避免个别 batch 取值恰好落在 [0,1] 内时误判。

    Returns:
        np.ndarray: 形状 (C,) 的 Dice 值；若输入带 batch 维，则在 B、H、W 三个维度上
            **合并统计**后返回 (C,)，即"整批合并"的 Dice，而不是只取第 0 个样本。
            需要逐切片的 Dice（论文表格的均值±标准差口径）时，请逐个样本调用本函数。
    """
    p = pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred)
    t = target.detach().cpu().numpy() if torch.is_tensor(target) else np.asarray(target)
    p = p.astype(np.float64, copy=False)

    p_ndim_orig = p.ndim
    if p.ndim == 3:                    # (C, H, W) -> (1, C, H, W)
        p = p[None]
    if p.ndim != 4:
        raise ValueError(f"pred 需为 (C,H,W) 或 (B,C,H,W)，收到 {tuple(pred.shape)}")

    t_ndim_orig = t.ndim
    if t.ndim == 2:                    # (H, W) 整数类别图 -> (1, H, W)
        t = t[None]
    elif t.ndim == 3 and t_ndim_orig == p_ndim_orig:
        t = t[None]                    # (C,H,W) 通道格式，与 pred 同步补 batch 维
    if t.ndim not in (3, 4):
        raise ValueError(f"target 需为 (H,W)/(B,H,W) 或 (C,H,W)/(B,C,H,W)，收到 {tuple(target.shape)}")

    n_classes = p.shape[1]
    if t.ndim == 4:                    # 通道格式：必须与 pred 同形
        if t.shape != p.shape:
            raise ValueError(f"pred 与 target 形状不一致: {p.shape} vs {t.shape}")
        t = t.astype(np.float64, copy=False)
        tc = (t >= 0.5).astype(np.float64)
    else:                              # 整数类别图：(B, H, W)
        if t.shape[0] != p.shape[0] or t.shape[1:] != p.shape[2:]:
            raise ValueError(
                f"pred 与 target 尺寸不一致: pred{b_batch(p)} vs target{t.shape}"
            )
        labels = np.rint(t.astype(np.float64))
        tc = np.stack([(labels == (c + 1)).astype(np.float64) for c in range(n_classes)], axis=1)

    if is_logit is None:
        is_logit = p.dtype.kind == "f" and not np.all((p == 0.0) | (p == 1.0))
    if is_logit:
        p = 1.0 / (1.0 + np.exp(-p))          # 数值稳定的 sigmoid

    pb = (p >= threshold).astype(np.float64)

    dims = (0, 2, 3)                          # 跨 batch 合并，避免只统计第 0 个样本
    inter = (pb * tc).sum(axis=dims)
    denom = pb.sum(axis=dims) + tc.sum(axis=dims)
    dice = np.where(denom > 0, 2.0 * inter / (denom + eps), 1.0)
    return dice.astype(np.float64)


def b_batch(p: np.ndarray) -> tuple:
    """错误信息里用于展示 pred 形状的小工具。"""
    return tuple(p.shape)


def aggregate_dice(per_slice: list[np.ndarray]) -> dict[str, float]:
    """把逐切片的 Dice 聚合成论文表格口径（各类别均值）。

    论文报告的是逐 B-scan Dice 的均值与标准差（标准差可达 ±27，说明患者间差异极大），
    因此这里对切片取均值，而不是先聚合成体积再算体积级 Dice。
    """
    if not per_slice:
        return {**{name: float("nan") for name in CLASS_NAMES}, "Mean": float("nan")}

    arr = np.stack(per_slice, axis=0)          # (N, C)
    out: dict[str, float] = {}
    for ci, name in enumerate(CLASS_NAMES):
        out[name] = float(np.mean(arr[:, ci]))
    out["Mean"] = float(np.mean(arr))          # 论文的 Mean 为三类别的简单平均
    return out


def dice_std(per_slice: list[np.ndarray]) -> dict[str, float]:
    """逐类别 Dice 的标准差，对齐论文表格中的 ± 数值。"""
    if not per_slice:
        return {name: float("nan") for name in CLASS_NAMES}
    arr = np.stack(per_slice, axis=0)
    return {name: float(np.std(arr[:, ci])) for ci, name in enumerate(CLASS_NAMES)}
