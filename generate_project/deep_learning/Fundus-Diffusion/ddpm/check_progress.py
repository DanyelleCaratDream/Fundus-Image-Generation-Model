# -*- coding: utf-8 -*-
"""一键查看 A/C/B 实验链进度。Windows 下双击 `查看实验进度.bat` 即可运行。"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# GBK 控制台打不出 ✅ 等字符时直接崩，这里降级为替换输出而不是报错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

HERE = Path(__file__).resolve().parent
COOL = "(含 --cool_sleep 休眠，所以每轮比纯训练慢约 50%)"

# (显示名, 目录, 总轮数, 说明, 是否已人工放弃)
EXPS = [
    ("D1 · 193张@256 base_dim64", "results_D1_combined239_bd64", 500, "★ 验证数据量假说", False),
    ("D2 · 193张@256 base_dim128", "results_D2_combined239_bd128", 300, "验证容量假说", False),
    ("B · 微调 A 域模型 @128", "results_B_severe128_finetune", 300, "已完成到1100轮", True),
    ("A · sev76@128 base_dim64", "results_A_severe128_scratch", 800, "分辨率不是主因", True),
    ("C · sev76@256 base_dim128", "results_C_severe256_bd128", 400, "改跑 B，已停", True),
]

# 日志超过这个秒数没更新，就认为该实验已停止（不是"进行中"）
STALE_SEC = 300


def read_log(p: Path) -> tuple[int, list[float], list[float], str]:
    """返回 (轮数, loss列表, 耗时列表, 最后一行)"""
    losses, times = [], []
    n = 0
    last = ""
    try:
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("Epoch "):
                    n += 1
                    last = line.strip()
                    try:
                        parts = line.split("loss=")[1]
                        losses.append(float(parts.split(",")[0]))
                        times.append(float(line.split("time=")[1].split("s")[0]))
                    except Exception:
                        pass
    except FileNotFoundError:
        pass
    return n, losses, times, last


def gpu_busy() -> bool:
    """GPU 上有没有计算进程——最可靠地判断'是不是真的有东西在训练'。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        return bool(out.stdout.strip())
    except Exception:
        return True   # 查不到就当忙，避免误报"已停止"


def main() -> int:
    print("=" * 74)
    print(f"  眼底生成实验链进度          {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 74)

    busy = gpu_busy()
    running = None
    # 只有"日志最新"的那个实验算运行中；其余比它老 10 分钟以上就认为已停
    mtimes = []
    for _, d, _, _, _ in EXPS:
        p = HERE / d / "logs" / "training.log"
        if p.exists():
            mtimes.append(os.path.getmtime(p))
    newest = max(mtimes) if mtimes else 0
    for name, d, total, purpose, abandoned in EXPS:
        log = HERE / d / "logs" / "training.log"
        if not (HERE / d).is_dir():
            print(f"  {name:30s}  未开始        （{purpose}）")
            continue
        n, losses, times, last = read_log(log)
        if n == 0:
            print(f"  {name:30s}  刚启动，日志未生成")
            continue
        done = min(n, total)
        pct = done * 100 // total
        # 判断是「进行中」还是「已停止」：看日志最后修改时间
        try:
            age = time.time() - os.path.getmtime(log)
        except OSError:
            age = 1e9
        if n >= total:
            state = "✅ 完成"
        elif abandoned:
            state = f"❌ 已放弃（{purpose}）"
        elif not busy:
            state = "⏹ 已停止（GPU 空闲）"
        elif (newest - os.path.getmtime(log)) > STALE_SEC:
            state = f"⏹ 已停止（落后最新日志 {(newest-os.path.getmtime(log))/60:.0f} 分钟）"
        else:
            state = "🏃 进行中"
            running = (name, n, total, times)
        # 用最近 20 轮的中位数估时，避开 checkpoint/预览造成的尖峰
        recent = sorted(times[-20:])
        med = recent[len(recent) // 2] if recent else 0
        print(f"  {name:30s}  {done:3d}/{total:<3d} ({pct:3d}%)  "
              f"loss={losses[-1]:.4f}  {med:.1f}s/轮  {state}")
        if running and running[0] == name and med > 0:
            rest = total - n
            print(f"  {'':30s}  → 预计还需 {rest * med / 60:.0f} 分钟")

    print("-" * 74)
    print(f"  提示：{COOL}")
    print("\n  输出位置：")
    for name, d, _, _, _ in EXPS:
        p = HERE / d
        if p.is_dir():
            print(f"    {d}/  （日志在 logs/training.log，预览图在 images/）")
    print("\n  完整日志路径（可直接粘进记事本或浏览器打开）：")
    for name, d, _, _, _ in EXPS:
        p = HERE / d / "logs" / "training.log"
        if p.exists():
            print(f"    {p}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
