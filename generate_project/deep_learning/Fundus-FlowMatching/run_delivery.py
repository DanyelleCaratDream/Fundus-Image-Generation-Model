#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""交付批次生成：CFM@256 无条件批量采样（final + lesion 两检查点各 200 张）

用途：Phase D 交付——生成 severe 新图池供作者筛选 76 张替换 train 里翻转副本。
每检查点 200 张（seed 200–399，与评估 60 张 seed 100–159 不重叠），Euler 100 NFE，
256px，条件走 conditions_combined_256（193 mask 循环）。
逐批落盘、可断点续跑（--resumable）：中途被杀后重跑同命令，已完成的图自动跳过。
SDEdit 精修留作后续对挑选图单独跑（CFM/infer/deliver_cfm.py → deliver_cfm256_s08.py）。

温度纪律：每检查点前 wait_temp(70)；生成不比训练持续但 200×100NFE 仍长载，超 70C 停 60s。
"""
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
PY = sys.executable
LOG = CFM.parent / "RUN_DELIVERY.log"
STATUS = CFM.parent / "DELIVERY_STATUS.json"


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(step, state):
    STATUS.write_text(json.dumps({"step": step, "state": state,
                                  "time": datetime.now().isoformat()},
                                 ensure_ascii=False, indent=2), encoding="utf-8")


def gpu_temp():
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            timeout=5, text=True).strip())
    except Exception:
        return 0


def wait_temp(limit=70):
    for _ in range(30):
        t = gpu_temp()
        if t <= limit:
            return
        log(f"  [temp] {t}C > {limit}C, wait 60s")
        time.sleep(60)


def run(cmd, cwd, name):
    wait_temp()
    log(f"  RUN {name}: {' '.join(str(c) for c in cmd[3:])}")
    with open(LOG, "a", encoding="utf-8") as fo:
        r = subprocess.run([str(c) for c in cmd], cwd=str(cwd), stdout=fo,
                           stderr=subprocess.STDOUT)
    log(f"  END {name}: exit={r.returncode}")
    return r.returncode


BATCHES = [
    ("final", CFM / "models/cfm256_final_model.pth", REPO / "eval_data/delivery_final"),
    ("lesion", CFM / "models/cfm256_lesion_final_model.pth", REPO / "eval_data/delivery_lesion"),
]


def main():
    LOG.write_text("", encoding="utf-8")
    log("DELIVERY START (2 ckpts x 200 imgs, seed 200-399, Euler 100 NFE)")
    for tag, ckpt, out in BATCHES:
        status(tag, "running")
        cmd = [PY, DDPM / "generate.py", "--checkpoint", ckpt,
               "--num_images", "200", "--img_size", "256", "--base_dim", "128",
               "--seed", "200", "--sampling_steps", "100", "--sampler", "euler",
               "--cond_path", CFM / "data/conditions_combined_256",
               "--output_dir", out, "--grid_size", "0", "--resumable"]
        rc = run(cmd, DDPM, f"deliver-{tag}")
        n = len(list(Path(out / "singles").glob("*.png"))) if (out / "singles").exists() else 0
        log(f"  [{tag}] {n} pngs in {out}/singles")
        status(tag, "done" if rc == 0 else "error")
    log("DELIVERY DONE")
    status("all", "done")


if __name__ == "__main__":
    main()
