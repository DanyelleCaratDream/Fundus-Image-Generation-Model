#!/usr/bin/env python3
"""经典测试时自适应基线（TENT / CoTTA）——作用于**下游分割模型**。

论文定位（决定了实现的作用对象，务必先读）
------------------------------------------
论文第 2 节把 TENT [24] / CoTTA [25] 归为 "early methods updated pre-trained model
weights during inference using target domain data and specialized loss functions"，
第 3 节将其列为对照方法。它们**不是图像重建方法**：
* 不产生重建图像，而是让已训练好的下游分割模型在目标域图像上在线自适应；
* 表格中 CoTTA/TENT 行的 GA、FID 列为 N/A —— 没有重建输出，自然无法计算 FID。

因此这里的作用对象是：**在 Spectralis 上训练好的分割模型 + 目标设备原始图像**
（与表格里的 Baseline 行完全相同的模型与切片），这样 Baseline / TENT / CoTTA 三者
的差异才唯一归因于适配策略本身。**不要**把它们接到重建结果上——那会变成另一种方法。

实现要点
--------
* TENT：只更新归一化层的仿射参数（γ/β），以预测熵为目标做在线优化。
* CoTTA：教师-学生 EMA + 随机参数回退（stochastic restore）+ 增强平均伪标签 +
  置信度筛选与类别均衡，以缓解长期自适应中的误差累积。
* 两者都让归一化层**使用测试批统计量**（TENT 原论文设定），并保证源域 running stats
  不被污染——否则同一份 checkpoint 在不同评测顺序下结果会漂移。
* 预测熵对多标签 sigmoid 头取"各通道二值熵之和"：这正是各通道独立伯努利分布的联合熵，
  与 TENT 在单标签 softmax 上的定义同构。
* 本文件不加载数据、不写结果，仅提供可测试的适配器；驱动脚本见 `eval_tta_seg.py`。
"""

from __future__ import annotations

import copy
import logging
import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

NORM_TYPES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d, nn.InstanceNorm1d,
)
_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


# --------------------------------------------------------------------------
# 参数与归一化层的选择
# --------------------------------------------------------------------------
def norm_layers(model: nn.Module) -> list[nn.Module]:
    """收集模型内所有归一化层。"""
    return [m for m in model.modules() if isinstance(m, NORM_TYPES)]


def affine_params(model: nn.Module) -> list[nn.Parameter]:
    """收集归一化层的仿射参数（γ/β）。无仿射的层直接跳过。"""
    out: list[nn.Parameter] = []
    for m in norm_layers(model):
        affine = getattr(m, "affine", None)
        if affine is None:
            affine = getattr(m, "elementwise_affine", False)
        if not affine:
            continue
        if getattr(m, "weight", None) is not None:
            out.append(m.weight)
        if getattr(m, "bias", None) is not None:
            out.append(m.bias)
    return out


def set_trainable(model: nn.Module, scope: str = "bn_affine") -> list[nn.Parameter]:
    """按 scope 冻结/解冻参数，返回需要优化的参数列表。

    Args:
        scope: "bn_affine" = 仅归一化层仿射参数（TENT 原设定）；
               "all" = 全部参数（CoTTA 原设定）。
    """
    if scope not in ("bn_affine", "all"):
        raise ValueError(f"未知 scope: {scope!r}（可选 bn_affine / all）")

    for p in model.parameters():
        p.requires_grad_(False)

    target = affine_params(model) if scope == "bn_affine" else list(model.parameters())
    for p in target:
        p.requires_grad_(True)

    if not target:
        raise RuntimeError(f"scope={scope} 未选中任何参数，模型结构可能不符合预期")
    n = sum(p.numel() for p in target)
    log.info("适配参数范围 scope=%s：%d 个张量 / %d 个参数", scope, len(target), n)
    return target


def set_norm_batch_stats(model: nn.Module, use_batch_stats: bool) -> None:
    """切换归一化层的统计量来源。

    use_batch_stats=True（TENT/CoTTA 的默认设定）：BatchNorm 走**测试批统计量**，
    同时关闭 running stats 的累积，源域统计量不被污染。
    use_batch_stats=False：完全沿用源域的 running stats。

    注意：只改归一化层的模式，**不**把整个模型设为 train()，
    以免重新打开 dropout / 其它训练态行为，让评测变得不可复现。
    """
    for m in norm_layers(model):
        if isinstance(m, _BN_TYPES) and use_batch_stats:
            m.train()
            m.track_running_stats = False   # 用批统计量且不回写 running stats
        else:
            m.eval()
            if isinstance(m, _BN_TYPES):
                m.track_running_stats = True


def binary_entropy(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """多标签 sigmoid 头的预测熵：各通道二值熵之和，再对 batch 取平均。

    各通道独立伯努利分布的联合熵即为各通道熵之和，与 TENT 在 softmax 上的定义同构。
    """
    p = torch.sigmoid(logits)
    ent = -(p * torch.log(p + eps) + (1.0 - p) * torch.log(1.0 - p + eps))
    return ent.sum(dim=1).mean()


# --------------------------------------------------------------------------
# 灰度 B-scan 的随机增强（CoTTA 的"增强平均伪标签"用）
# --------------------------------------------------------------------------
def _gaussian_kernel(size: int, sigma: float, device, dtype) -> torch.Tensor:
    ax = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)


def gaussian_blur(x: torch.Tensor, sigma: float, size: int = 5) -> torch.Tensor:
    """按通道分组的可微高斯模糊（reflect padding，避免边界出现黑边）。"""
    if sigma <= 0:
        return x
    k = _gaussian_kernel(size, sigma, x.device, x.dtype)
    k = k[None, None].repeat(x.shape[1], 1, 1, 1)
    pad = size // 2
    return F.conv2d(F.pad(x, (pad,) * 4, mode="reflect"), k, groups=x.shape[1])


def small_affine(x: torch.Tensor, angle_deg: float, translate: tuple[float, float],
                 scale: float) -> torch.Tensor:
    """小幅仿射（旋转/平移/缩放），尺度控制在小范围内以保留病灶位置。"""
    b = x.shape[0]
    theta = torch.zeros(b, 2, 3, device=x.device, dtype=x.dtype)
    a = angle_deg * math.pi / 180.0
    cos, sin = math.cos(a), math.sin(a)
    theta[:, 0, 0] = cos / scale
    theta[:, 0, 1] = -sin / scale
    theta[:, 1, 0] = sin / scale
    theta[:, 1, 1] = cos / scale
    theta[:, 0, 2] = translate[0]
    theta[:, 1, 2] = translate[1]
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                         align_corners=False)


def grayscale_augment(x: torch.Tensor, gen: torch.Generator, strength: float = 1.0) -> torch.Tensor:
    """OCT 灰度图的随机增强：亮度/对比度抖动 + 高斯模糊 + 小幅仿射。

    原版 CoTTA 使用 color jitter / Gaussian blur / random affine（RGB 自然图像）。
    OCT 为单通道灰度图，色相/饱和度无意义，故只保留亮度与对比度项，
    并把仿射幅度限制在小范围内以避免破坏细小病灶——该替换已记录为配置偏离。
    """
    def u(lo: float, hi: float) -> float:
        return float(torch.empty(1).uniform_(lo, hi, generator=gen).item())

    out = x
    if float(torch.rand(1, generator=gen).item()) < 0.8:
        a, b = 1.0 + u(-0.2, 0.2) * strength, u(-0.1, 0.1) * strength
        out = out * a + b
    if float(torch.rand(1, generator=gen).item()) < 0.5:
        out = gaussian_blur(out, sigma=u(0.5, 1.5) * strength)
    if float(torch.rand(1, generator=gen).item()) < 0.5:
        out = small_affine(out, angle_deg=u(-5, 5) * strength,
                           translate=(u(-0.05, 0.05) * strength, u(-0.05, 0.05) * strength),
                           scale=1.0 + u(-0.05, 0.05) * strength)
    # 亮度/对比度抖动可能把取值推出原始动态范围，夹回输入自身的 [min, max]，
    # 避免把分布外数值喂给网络（与 torchvision 的 color jitter 会截断的做法一致）
    return out.clamp(float(x.min()), float(x.max()))

# --------------------------------------------------------------------------
# TENT：在线熵最小化
# --------------------------------------------------------------------------
@dataclass
class TentConfig:
    """TENT 超参。论文未给出 TENT 的具体取值（仅称经 grid search 选定），
    这里采用原论文 ImageNet-C 的默认设定，并在报告中记录。"""

    lr: float = 1e-3
    momentum: float = 0.9
    steps: int = 1                    # 每个测试批的更新步数（原论文 N=1）
    scope: str = "bn_affine"          # 原论文只更新 BN 仿射参数
    use_batch_stats: bool = True      # 用测试批统计量做归一化


class TentAdapter:
    """TENT：固定其它参数，只最小化预测熵来更新归一化层仿射参数。

    预测阶段同样使用测试批统计量（原论文设定），因此**batch size 必须 >1**，
    否则批统计量退化，结果不可信。
    """

    name = "tent"

    def __init__(self, model: nn.Module, cfg: TentConfig | None = None) -> None:
        self.cfg = cfg or TentConfig()
        self.model = model
        self.source_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.params = set_trainable(model, self.cfg.scope)
        self.optimizer = torch.optim.SGD(self.params, lr=self.cfg.lr, momentum=self.cfg.momentum)
        self.model.eval()
        set_norm_batch_stats(self.model, self.cfg.use_batch_stats)

    def describe(self) -> dict:
        return {"name": self.name, **asdict(self.cfg)}

    def reset(self) -> None:
        """episodic 模式：参数与优化器状态都回到源域初始值。"""
        self.model.load_state_dict(self.source_state)
        self.params = set_trainable(self.model, self.cfg.scope)
        self.optimizer = torch.optim.SGD(self.params, lr=self.cfg.lr,
                                         momentum=self.cfg.momentum)

    @torch.enable_grad()
    def adapt(self, x: torch.Tensor) -> dict:
        """在无标签目标域批次上做 steps 步熵最小化，返回熵值曲线。"""
        self.model.eval()
        set_norm_batch_stats(self.model, self.cfg.use_batch_stats)

        curve: list[float] = []
        for _ in range(self.cfg.steps):
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = binary_entropy(logits)
            loss.backward()
            self.optimizer.step()
            curve.append(float(loss.detach().item()))
        return {"entropy_curve": curve}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """用适配后的模型预测（归一化层仍走测试批统计量）。"""
        self.model.eval()
        set_norm_batch_stats(self.model, self.cfg.use_batch_stats)
        return self.model(x)


# --------------------------------------------------------------------------
# CoTTA：教师-学生 EMA + 随机回退
# --------------------------------------------------------------------------
@dataclass
class CottaConfig:
    """CoTTA 超参。

    取值来源：EMA 0.999、随机回退概率 0.01、置信度阈值 0.5、队列长度 100 取自
    CoTTA 原论文默认值；学习率与动量沿用 SGD 的常规设定。论文只说明"经 grid search
    选定"，未给具体数值，故这些取值需在报告中记为实现选择。
    """

    lr: float = 1e-3
    momentum: float = 0.9
    steps: int = 1
    ema_alpha: float = 0.999
    restore_prob: float = 0.01
    n_aug: int = 2                    # 增强个数（伪标签由多视图平均得到）
    conf_threshold: float = 0.5       # 置信度阈值 τ
    queue_size: int = 100             # 类别均衡队列长度（批级条目）
    scope: str = "all"                # 原论文更新全部参数
    use_batch_stats: bool = True
    num_classes: int = 3
    seed: int = 42


class CottaAdapter:
    """CoTTA：教师模型产生增强平均伪标签，学生模型更新，教师 EMA 跟随，
    并按概率 p 把教师的一部分参数随机回退到源域值，抑制误差累积。

    预测使用**教师**模型（原论文设定）。
    """

    name = "cotta"

    def __init__(self, model: nn.Module, cfg: CottaConfig | None = None) -> None:
        self.cfg = cfg or CottaConfig()
        self.student = model
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()
        set_norm_batch_stats(self.teacher, self.cfg.use_batch_stats)

        self.source_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.source_params = {n: p.detach().clone() for n, p in self.teacher.named_parameters()}

        self.params = set_trainable(self.student, self.cfg.scope)
        self.optimizer = torch.optim.SGD(self.params, lr=self.cfg.lr, momentum=self.cfg.momentum)

        # 每个类别一个队列；条目为"整批"级别，只保存被选中类别的硬伪标签
        self.queues: list[deque] = [deque() for _ in range(self.cfg.num_classes)]
        device = next(self.student.parameters()).device
        self.gen_cpu = torch.Generator()                      # 增强用（CPU 标量）
        self.gen_cpu.manual_seed(self.cfg.seed)
        self.gen_dev = torch.Generator(device=device)         # 随机回退用（与参数同设备）
        self.gen_dev.manual_seed(self.cfg.seed)

    def describe(self) -> dict:
        return {"name": self.name, **asdict(self.cfg)}

    def reset(self) -> None:
        """episodic 模式：学生与教师都回到源域初始值，清空队列。"""
        self.student.load_state_dict(self.source_state)
        self.teacher.load_state_dict(self.source_state)
        self.teacher.eval()
        set_norm_batch_stats(self.teacher, self.cfg.use_batch_stats)
        self.params = set_trainable(self.student, self.cfg.scope)
        self.optimizer = torch.optim.SGD(self.params, lr=self.cfg.lr,
                                         momentum=self.cfg.momentum)
        self.queues = [deque() for _ in range(self.cfg.num_classes)]

    @torch.no_grad()
    def _teacher_probs(self, x: torch.Tensor) -> torch.Tensor:
        """对多个增强视图取教师预测的平均，得到更稳的伪标签。"""
        acc = None
        for _ in range(max(1, self.cfg.n_aug)):
            aug = grayscale_augment(x, self.gen_cpu)
            p = torch.sigmoid(self.teacher(aug))
            acc = p if acc is None else acc + p
        return acc / max(1, self.cfg.n_aug)

    @torch.enable_grad()
    def adapt(self, x: torch.Tensor) -> dict:
        """一步 CoTTA 更新，返回本步的元信息（类别/置信度/是否跳过）。"""
        self.teacher.eval()
        set_norm_batch_stats(self.teacher, self.cfg.use_batch_stats)

        probs = self._teacher_probs(x)
        hard = probs >= self.cfg.conf_threshold

        # 逐类置信度：该类被判为前景的像素上概率的均值
        confs: list[float] = []
        for c in range(self.cfg.num_classes):
            m = hard[:, c]
            confs.append(float(probs[:, c][m].mean()) if bool(m.any()) else -1.0)

        c_max = int(np.argmax(confs))
        info = {"class": c_max, "conf": confs[c_max], "skipped": False}
        if confs[c_max] < 0:
            # 本批没有任何类别出现超阈值预测：不更新，避免把纯噪声当成伪标签
            info["skipped"] = True
            self._ema_update()
            self._stochastic_restore()
            return info

        item = (x.detach(), hard[:, c_max].float(), c_max)
        q = self.queues[c_max]
        train_items = [item]
        if len(q) >= self.cfg.queue_size:
            train_items.append(q.popleft())     # 队列满则取出最旧的一条一起训练
        q.append(item)

        # 学生更新：对每个条目只监督其被选中的那个类别通道
        self.student.eval()
        set_norm_batch_stats(self.student, self.cfg.use_batch_stats)
        total = None
        for xi, yi, ci in train_items:
            logits = self.student(xi)[:, ci]
            loss = F.binary_cross_entropy_with_logits(logits, yi)
            total = loss if total is None else total + loss
        total = total / len(train_items)

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.optimizer.step()
        info["loss"] = float(total.detach().item())

        self._ema_update()
        self._stochastic_restore()
        return info

    @torch.no_grad()
    def _ema_update(self) -> None:
        a = self.cfg.ema_alpha
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
            tp.mul_(a).add_(sp.detach(), alpha=1.0 - a)

    @torch.no_grad()
    def _stochastic_restore(self) -> None:
        """按概率 p 把教师参数逐元素回退到源域值（原论文的抗误差累积机制）。"""
        p = self.cfg.restore_prob
        if p <= 0:
            return
        for name, tp in self.teacher.named_parameters():
            src = self.source_params[name]
            mask = torch.rand(tp.shape, generator=self.gen_dev, device=tp.device) < p
            tp.copy_(torch.where(mask, src, tp))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """用教师模型预测（原论文设定）。"""
        self.teacher.eval()
        set_norm_batch_stats(self.teacher, self.cfg.use_batch_stats)
        return self.teacher(x)
