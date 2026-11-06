#!/usr/bin/env python3
"""把评测 JSON 汇总成与论文 Table 1 同构的 markdown 对照表。

输入（均来自 `eval_dice.py` / `eval_tta_seg.py` 的输出，缺哪个就留空行）
------------------------------------------------------------------
* `baseline.json`   分割模型直接推理目标设备原始图像（无适配）
* `tent.json`       TENT 在线适配
* `cotta.json`      CoTTA 在线适配
* `ours.json`       推理 flow matching 重建结果（本方法）
* `crosscheck.json` `eval_tta_seg.py --mode none`，用于与 baseline 互证

行序按论文表格：Baseline → TENT → CoTTA → Ours。

用法
----
    python eval/summarize.py --eval-dir <实验目录>/eval --device cirrus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 论文 Table 1 的报告值（Ours_uncond. 行），用于并排对照
PAPER = {
    "cirrus": {"IRF": 58.8, "SRF": 66.3, "PED": 50.6, "Mean": 58.6},
    "topcon": {"IRF": 51.6, "SRF": 52.2, "PED": 49.2, "Mean": 51.0},
}

# (显示名, 文件名)——顺序即论文表格顺序
ROWS = [
    ("Baseline（无适配）", "baseline.json"),
    ("TENT [24]", "tent.json"),
    ("CoTTA [25]", "cotta.json"),
    ("Ours_uncond.（本复现）", "ours.json"),
]
OURS_LABEL = ROWS[-1][0]
KEYS = ("IRF", "SRF", "PED", "Mean")


def load(eval_dir: Path, fname: str) -> dict | None:
    p = eval_dir / fname
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"⚠️  {fname} 解析失败，按缺失处理: {exc}", file=sys.stderr)
        return None


def fmt(value: float | None, std: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f} ± {std:.1f}" if std is not None else f"{value:.1f}"


def build(eval_dir: Path, device: str) -> str:
    lines = [
        f"# 评测汇总：{device} → Spectralis（复现值）",
        "",
        "| 方法 | " + " | ".join(KEYS) + " | 体积数 | 切片数 |",
        "|" + "---|" * (len(KEYS) + 3),
    ]

    data: dict[str, dict] = {}
    missing: list[str] = []
    for label, fname in ROWS:
        d = load(eval_dir, fname)
        if d is None:
            missing.append(fname)
            lines.append(f"| {label} | " + " | ".join(["—"] * len(KEYS)) + " | — | — |")
            continue
        data[label] = d
        dice, std = d.get("dice", {}), d.get("dice_std", {})
        cells = [fmt(dice.get(k), std.get(k)) for k in KEYS]
        lines.append(f"| {label} | " + " | ".join(cells) +
                     f" | {d.get('num_volumes', '—')} | {d.get('num_slices', '—')} |")

    # 两条独立实现路径的一致性检查（自测中已确认应逐位相同）
    cc = load(eval_dir, "crosscheck.json")
    if cc is not None and "Baseline（无适配）" in data:
        a, b = cc.get("dice", {}), data["Baseline（无适配）"].get("dice", {})
        common = [k for k in KEYS if a.get(k) is not None and b.get(k) is not None]
        if common:
            diff = max(abs(a[k] - b[k]) for k in common)
            note = ("✅ 逐位一致" if diff < 1e-9 else
                    f"❌ 存在差异（最大 {diff:.3e}）——后续 TENT/CoTTA 的差值不可信，请排查")
            lines += ["",
                      f"**交叉校验**（`eval_tta_seg.py --mode none` vs `eval_dice.py --raw-dir`）：{note}"]

    if missing:
        lines += ["", f"> ⚠️ 缺失 {len(missing)} 个结果文件：{', '.join(missing)}"]

    # 与论文报告值并排
    ours = data.get(OURS_LABEL, {}).get("dice", {})
    paper = PAPER.get(device, {})
    if ours or paper:
        lines += ["", "## 与论文对照（Table 1，Ours_uncond. 行）", "",
                  "| 指标 | 论文报告值 | 本复现值 | 差值 |", "|---|---|---|---|"]
        for k in KEYS:
            pv, ov = paper.get(k), ours.get(k)
            if pv is None and ov is None:
                continue
            lines.append(
                f"| {k} | {'—' if pv is None else f'{pv:.1f}'} "
                f"| {'—' if ov is None else f'{ov:.1f}'} "
                f"| {'—' if (pv is None or ov is None) else f'{ov - pv:+.1f}'} |"
            )

    lines += [
        "",
        "> 说明：论文的 ± 为标准差（逐 B-scan 口径）。复现值受迭代预算、下游分割模型的训练细节",
        "> （论文未给出）与数据预处理影响，绝对值不要求精确等值；方法的可验证论断是**相对关系**：",
        "> (a) Ours 优于 Baseline 与 FM 消融；(b) 无条件优于有条件；(c) 对 s_target 不敏感。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="汇总评测 JSON 为 markdown 对照表")
    ap.add_argument("--eval-dir", required=True, type=Path)
    ap.add_argument("--device", required=True, choices=sorted(PAPER))
    ap.add_argument("--out", type=Path, default=None, help="默认写入 <eval-dir>/summary.md")
    ap.add_argument("--print-only", action="store_true", help="只打印，不写文件（用于自检）")
    args = ap.parse_args()

    if not args.eval_dir.is_dir():
        print(f"❌ 目录不存在: {args.eval_dir}", file=sys.stderr)
        return 2

    text = build(args.eval_dir, args.device)
    print(text, end="")

    if not args.print_only:
        out = args.out or (args.eval_dir / "summary.md")
        out.write_text(text, encoding="utf-8")
        print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
