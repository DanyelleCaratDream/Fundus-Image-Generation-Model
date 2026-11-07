#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""修正管线两臂：C1'（λ=0）→ C2'（λ=8）顺序训练 + 自动评估

**仅在导师确认后运行**。两臂均从零、320 轮、同 seed 42、同步增强；唯一差异 λ。
每臂前等待 GPU 空闲且温度 ≤70°C；训练器自带每 25 步温度守护。

用法
    python run_clean_arms.py --dry-run     # 只打印命令
    python run_clean_arms.py               # 正式执行（约 19h）
    python run_clean_arms.py --skip-eval
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(r"D:/AI_Model_Project_for_Fundus_Color_Images")
DL = REPO / "generate_project/deep_learning"
CFM = DL / "Fundus-FlowMatching/CFM"
DDPM = DL / "Fundus-Diffusion/ddpm"
FLOW = CFM.parent
PY = sys.executable
LOG = FLOW / "RUN_CLEAN_ARMS.log"
STATUS = FLOW / "CLEAN_ARMS_STATUS.json"

ARMS = [
    ("cfm128_hrsevere_pilot", "cfm128_hrsevere_pilot", 128),
    ("cfm256_clean_ctrl", "cfm256_clean_ctrl", 256),
    ("cfm256_clean_exudate", "cfm256_clean_exudate", 256),
]


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(step, state):
    STATUS.write_text(json.dumps({"step": step, "state": state,
                                  "time": datetime.now().isoformat()},
                                 ensure_ascii=False, indent=2), encoding="utf-8")


def gpu_mem():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                     "--format=csv,noheader,nounits"], timeout=5, text=True)
        return int(o.strip().splitlines()[0])
    except Exception:
        return 0


def gpu_temp():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu",
                                            "--format=csv,noheader"], timeout=5, text=True).strip())
    except Exception:
        return 0


def wait_ready(limit_min=30):
    ok = 0
    t0 = time.time()
    while time.time() - t0 < limit_min * 60:
        if gpu_mem() < 1500 and gpu_temp() <= 70:
            ok += 1
            if ok >= 3:
                return True
        else:
            ok = 0
        time.sleep(10)
    return False


def run(cmd, cwd, name):
    log(f"RUN {name}: {' '.join(str(c) for c in cmd[:6])} ...")
    with open(LOG, "a", encoding="utf-8") as fo:
        r = subprocess.run([str(c) for c in cmd], cwd=str(cwd), stdout=fo,
                           stderr=subprocess.STDOUT)
    log(f"END {name}: exit={r.returncode}")
    return r.returncode


def eval_arm(tag, dry, img_size=256):
    ck = CFM / f"results/{tag}/models/final_model.pth"
    out = REPO / f"eval_data/{tag}"
    gen = [PY, DDPM / "generate.py", "--checkpoint", ck, "--num_images", "60",
           "--img_size", str(img_size), "--base_dim", "128", "--seed", "100",
           "--sampling_steps", "50", "--sampler", "euler",
           "--cond_path", CFM / f"data/conditions_hrsevere_61_{img_size}",
           "--output_dir", out, "--grid_size", "0"]
    common = [PY, REPO / "eval/metrics_common.py", "--real", REPO / "eval_data/real_severe_train",
              "--fake", out / "singles", "--img_size", "256", "--device", "cuda", "--json"]
    fundus = [PY, REPO / "eval/metrics_fundus.py", "--real", REPO / "eval_data/real_severe_train",
              "--fake", out / "singles", "--model", tag, "--img_size", "256", "--device", "cuda",
              "--cond_path", CFM / "data/conditions_hrsevere_61_256", "--json"]
    if dry:
        for c in (gen, common, fundus):
            print("  ", " ".join(str(x) for x in c))
        return 0
    rc = run(gen, DDPM, f"gen-{tag}")
    if rc != 0:
        return rc
    import os
    env = dict(os.environ); env["PYTHONIOENCODING"] = "utf-8"
    with open(REPO / f"eval_data/{tag}_metrics.json", "w", encoding="utf-8") as fo:
        rc1 = subprocess.run([str(x) for x in common], cwd=str(REPO), stdout=fo,
                             stderr=subprocess.DEVNULL, env=env).returncode
    with open(REPO / f"eval_data/{tag}_fundus_metrics.json", "w", encoding="utf-8") as fo:
        rc2 = subprocess.run([str(x) for x in fundus], cwd=str(REPO), stdout=fo,
                             stderr=subprocess.DEVNULL, env=env).returncode
    return rc1 or rc2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()

    if not a.dry_run:
        LOG.write_text("", encoding="utf-8")
        log("CLEAN ARMS START (pilot 128 + 2 arms 256, 320 epochs, from scratch)")

    for tag, preset, img_size in ARMS:
        train = [PY, CFM / "train/train_cfm.py", "--preset", preset]
        if a.dry_run:
            print(f"[arm] {tag} ({img_size}px)")
            print("  ", " ".join(str(x) for x in train))
            if not a.skip_eval:
                eval_arm(tag, dry=True, img_size=img_size)
            continue
        status(tag, "running")
        if not wait_ready():
            log(f"[{tag}] GPU not ready in 30 min; abort")
            status(tag, "error")
            continue
        rc = run(train, CFM, f"train-{tag}")
        if rc == 0 and not a.skip_eval:
            rc = eval_arm(tag, dry=False, img_size=img_size)
        status(tag, "done" if rc == 0 else "error")

    if not a.dry_run:
        log("CLEAN ARMS DONE")
        status("all", "done")


if __name__ == "__main__":
    main()
