#!/usr/bin/env python3
"""对官方仓库副本施加**最小、可审计、幂等**的补丁。

设计原则
--------
1. **默认不改变官方行为**：所有新增参数都有"等于官方语义"的默认值；不传参数时，
   代码路径与官方 commit `c2f1190` 逐行一致。
2. **幂等**：重复执行不会重复插入；已打过补丁会跳过并报告。
3. **可审计**：每处改动都在 `PATCHES` 中登记编号与理由，供复现报告直接引用。
4. **失败即停**：凡目标片段未找到，立即报错并列出该文件当前内容片段，绝不静默跳过。

用法：
    python patch_repo.py <仓库根目录> [--check]
    # --check 只检查补丁是否已应用，不写盘
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 补丁登记表
# ---------------------------------------------------------------------------
PATCHES: list[dict] = [
    {
        "id": "P1",
        "file": "src/train.py",
        "title": "可选梯度累积（train_parameters.grad_accum_steps，默认 1）",
        "reason": (
            "论文用 batch size 16 训练，但其硬件为 RTX A6000 48GB。本机为 RTX 4090 24GB，"
            "实测 224x224 下 batch=16 必然 OOM（batch=8 峰值 14.5GB）。"
            "为在 STEP 数不变的前提下保持与论文相同的有效 batch，加入梯度累积："
            "batch=8 x grad_accum_steps=2 == batch=16。"
            "损失为 torch.mean((v_hat-ut)^2)，micro-batch 等大，故 (loss/2) 反传再累加"
            "等价于对有效 batch 取均值。"
            "默认值 1 时行为与官方实现完全一致。"
        ),
        "old": """    # Iterate DataLoader
    for step in tqdm(range(start_step, cfg["train_parameters"]["max_iterations"])):
        optim.zero_grad()
                
        # Draw next batch, reset iterator at end of epoch
        try:
            x0, x1 = next(train_iter)
        except StopIteration:
            train_iter  = iter(train_dataloader)
            x0, x1     = next(train_iter)

        x0 = x0.to(device)
        x1 = x1.to(device)

        # Compute the loss. Class also performs prediction and estimation of the ground truth velocity field.
        loss    = loss_func(
            flow_matcher=flow_matcher,
            net=net_model,
            x0=x0,
            x1=x1,
        )
        writer.add_scalar("loss", loss, step)

        # Update
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_model.parameters(), cfg["train_parameters"]["grad_clip"])""",
        "new": """    # [P1] 可选梯度累积：grad_accum_steps=1 时与官方实现逐行等价
    grad_accum_steps = int(cfg["train_parameters"].get("grad_accum_steps", 1))
    log.info(f"Gradient accumulation steps: {grad_accum_steps} "
             f"(effective batch = {cfg['train_parameters']['train_batch_size']} x {grad_accum_steps})")

    # Iterate DataLoader
    for step in tqdm(range(start_step, cfg["train_parameters"]["max_iterations"])):
        optim.zero_grad()

        accumulated_loss = 0.0
        for _micro in range(grad_accum_steps):
            # Draw next batch, reset iterator at end of epoch
            try:
                x0, x1 = next(train_iter)
            except StopIteration:
                train_iter  = iter(train_dataloader)
                x0, x1     = next(train_iter)

            x0 = x0.to(device)
            x1 = x1.to(device)

            # Compute the loss. Class also performs prediction and estimation of the ground truth velocity field.
            loss    = loss_func(
                flow_matcher=flow_matcher,
                net=net_model,
                x0=x0,
                x1=x1,
            )
            accumulated_loss += float(loss.item())
            (loss / grad_accum_steps).backward()

        loss = accumulated_loss / grad_accum_steps
        writer.add_scalar("loss", loss, step)

        # Update
        torch.nn.utils.clip_grad_norm_(net_model.parameters(), cfg["train_parameters"]["grad_clip"])""",
    },
    {
        "id": "P2",
        "file": "preferences/train_parameters/param_fm.yaml",
        "title": "新增 grad_accum_steps 配置项（默认 1）",
        "reason": "为 P1 提供配置入口；默认 1 表示与官方行为一致。",
        "old": """grad_clip: 1.0
ema_decay: 0.9999""",
        "new": """grad_clip: 1.0
ema_decay: 0.9999
grad_accum_steps: 1                 # [P2] 梯度累积步数；1 = 官方行为。batch 8 x 2 等价于论文的 batch 16""",
    },
    {
        "id": "P3",
        "file": "src/test.py",
        "title": "可选 ODE 起点模式 test_parameters.ode_start（默认 from_zero = 官方行为）",
        "reason": (
            "论文正文与 README 均称\"ODE 从 s_target 处开始积分\"，但官方 test.py 把 s_target "
            "仅用于挑选匹配的目标直方图，积分区间恒为 linspace(0, 1, num_integration_steps)。"
            "为验证两种设定哪种对应论文报告结果，加入开关；"
            "默认 from_zero 表示与官方代码逐行等价。"
            "详见 repro/notes/code_semantics.md 第 1 节。"
        ),
        "old": """                # Generate trajectory
                inference_trajectory = ode_solver.solve(
                    x=slice_d,
                    t_span=torch.linspace(0, 1, num_timesteps, device=device),
                )""",
        "new": """                # Generate trajectory
                # [P3] ode_start="from_zero"（默认）完全复刻官方行为：恒从 t=0 积分到 t=1；
                #      ode_start="from_s_target" 才是论文/README 文字描述的"从 s_target 起步"。
                _ode_start = cfg["test_parameters"].get("ode_start", "from_zero")
                if _ode_start == "from_s_target" and num_references > 0:
                    _t_span = torch.linspace(0, 1, num_timesteps, device=device)[s_target:]
                elif _ode_start == "from_zero":
                    _t_span = torch.linspace(0, 1, num_timesteps, device=device)
                else:
                    raise ValueError(f"Unknown ode_start >> {_ode_start} <<, expected 'from_zero' or 'from_s_target'")
                inference_trajectory = ode_solver.solve(
                    x=slice_d,
                    t_span=_t_span,
                )""",
    },
    {
        "id": "P4",
        "file": "preferences/test_parameters/test_param_fm.yaml",
        "title": "新增 ode_start 配置项（默认 from_zero）",
        "reason": "为 P3 提供配置入口；默认 from_zero 表示与官方行为一致。",
        "old": """num_reference_trajectories: 50
s_target: 0.8""",
        "new": """num_reference_trajectories: 50
s_target: 0.8
ode_start: "from_zero"              # [P4] from_zero = 官方代码行为；from_s_target = 论文/README 文字描述""",
    },
]


def apply_patch(repo: Path, spec: dict, check_only: bool) -> str:
    """施加单个补丁；返回状态字符串。

    幂等性判据用补丁自己的唯一标记 `[Pn]`，而不是"新旧片段是否互为子串"——
    后者对 YAML 这类"新片段包含旧片段"的情形会失效，导致重复插入。
    """
    path = repo / spec["file"]
    if not path.exists():
        raise SystemExit(f"[{spec['id']}] 目标文件不存在: {path}")

    text = path.read_text(encoding="utf-8")
    marker = f"[{spec['id']}]"

    # 先用标记判断是否已打过；已打过的文件里应恰好出现一次标记
    n_marker = text.count(marker)
    if n_marker > 1:
        raise SystemExit(
            f"[{spec['id']}] 检测到 {n_marker} 处补丁标记（应为 1 处），文件可能被重复打补丁: {path}\n"
            f"请手工修复后再运行。"
        )
    if n_marker == 1:
        return "已应用（跳过）"
    if spec["old"] not in text:
        head = "\n".join(text.splitlines()[:5])
        raise SystemExit(
            f"[{spec['id']}] 未找到待替换片段，无法安全打补丁。\n"
            f"文件: {path}\n"
            f"请确认仓库是否为 commit c2f1190（片段可能已随版本变化）。\n"
            f"文件开头:\n{head}"
        )

    if check_only:
        return "待应用"

    path.write_text(text.replace(spec["old"], spec["new"], 1), encoding="utf-8")
    return "已应用"


def main() -> int:
    ap = argparse.ArgumentParser(description="对官方仓库副本施加最小补丁")
    ap.add_argument("repo", type=Path, help="仓库根目录")
    ap.add_argument("--check", action="store_true", help="只检查，不写盘")
    args = ap.parse_args()

    if not (args.repo / "src" / "train.py").exists():
        raise SystemExit(f"看起来不是 TTA-Flow 仓库根目录: {args.repo}")

    print(f"仓库: {args.repo.resolve()}")
    print(f"模式: {'只检查' if args.check else '施加补丁'}\n")
    for spec in PATCHES:
        status = apply_patch(args.repo, spec, args.check)
        print(f"  [{spec['id']}] {status:10s} {spec['file']}")
        print(f"        {spec['title']}")
    print("\n✅ 全部补丁处理完毕")
    print("   默认配置下（不传新参数）代码行为与官方 c2f1190 一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
