#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""流水线结束后的补跑队列：featC 评估 → HM → 检查点扫描 → Tiger 零样本

背景：2026-10-24 首轮流水线里 featC 评估（.cuda() bug）与 HM（no_grad 缺失导致 OOM）
失败；此处修复后补跑，并追加检查点扫描（诊断用，不改主结果）。
"""
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO = Path(r"D:/AI_Model_Project_for_Fundus_Color_Images")
DL = REPO / "generate_project/deep_learning"
CFM = DL / "Fundus-FlowMatching/CFM"
DDPM = DL / "Fundus-Diffusion/ddpm"
TIGER = REPO / "generate_project/pretrained/TextGuidedDiffusion"
FLOW = CFM.parent
PY = sys.executable
PIPE_STATUS = FLOW / "RUN_PIPELINE_STATUS.json"
STATUS = FLOW / "POST_STATUS.json"
LOG = FLOW / "RUN_POST.log"


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(step, state):
    STATUS.write_text(json.dumps({"step": step, "state": state,
                                  "time": datetime.now().isoformat()},
                                 ensure_ascii=False, indent=2), encoding="utf-8")


def run(cmd, cwd, name):
    log(f"  RUN {name}")
    with open(LOG, "a", encoding="utf-8") as fo:
        r = subprocess.run([str(c) for c in cmd], cwd=str(cwd), stdout=fo,
                           stderr=subprocess.STDOUT)
    log(f"  END {name}: exit={r.returncode}")
    return r.returncode


def safe(name, fn):
    status(name, "running")
    try:
        fn(); status(name, "done")
    except Exception:
        log(f"STEP {name} ERROR\n" + traceback.format_exc())
        status(name, "error")


def wait_pipeline():
    while True:
        try:
            d = json.loads(PIPE_STATUS.read_text(encoding="utf-8"))
        except Exception:
            d = {}
        if d.get("step") == "all" and d.get("state") in ("done", "error"):
            return
        time.sleep(60)


def main():
    LOG.write_text("", encoding="utf-8")
    log("POST QUEUE START (waiting pipeline)")
    wait_pipeline()
    log("pipeline done; start repairs")

    safe("eval_featC", lambda: run([
        PY, CFM / "tools/eval_feat_model.py",
        "--ckpt", CFM / "models/cfm128_featC_final_model.pth",
        "--conds", CFM / "data/handcrafted/conds_severe.npy",
        "--tag", "cfm128_featC", "--num", "60",
        "--cond", CFM / "data/conditions_combined_128"], CFM, "eval-featC"))

    safe("tta_hm", lambda: run([
        PY, CFM / "tools/tta_hist_match.py",
        "--ckpt", CFM / "models/cfm256_final_model.pth",
        "--src", REPO / "fundus/HRSG_4_758_sq256/train/severe",
        "--cond", CFM / "data/conditions_combined_256",
        "--num_ref", "8", "--s_target", "0.5", "--steps", "100",
        "--limit", "8", "--out", CFM / "results/tta_hm"], DDPM, "tta_hm"))

    scans = [
        ("cfm128", Path(r"D:/tmp/_archive_cfm_ckpts_20261023/cfm128"), [800, 900, 1000, 1100], 128,
         CFM / "data/conditions_combined_128", None),
        ("cfm256", Path(r"D:/tmp/_archive_cfm_ckpts_20261023/cfm256"), [1150, 1250, 1350, 1420], 256,
         CFM / "data/conditions_combined_256", None),
        ("cfm128_featC", CFM / "results/cfm128_featC/models", [800, 900, 1000, 1100], 128,
         CFM / "data/conditions_combined_128", CFM / "data/handcrafted/conds_severe.npy"),
        ("cfm128_align", CFM / "results/cfm128_align/models", [850, 950, 1100], 128,
         CFM / "data/conditions_combined_128", None),
        ("cfm128_text", CFM / "results/cfm128_text/models", [850, 950, 1100], 128,
         CFM / "data/conditions_combined_128", None),
        ("cfm128_notime", CFM / "results/cfm128_notime/models", [850, 950, 1100], 128,
         CFM / "data/conditions_combined_128", None),
        ("cfm128_featA", CFM / "results/cfm128_featA/models", [850, 950, 1100], 128,
         CFM / "data/conditions_combined_128", CFM / "data/retfound_cfp/conds_severe.npy"),
    ]
    for tag, ckdir, eps, size, cond, feat in scans:
        cmd = [PY, CFM / "tools/scan_checkpoints.py", "--ckpt_dir", ckdir,
               "--epochs", *eps, "--img_size", size, "--cond", cond, "--tag", tag]
        if feat:
            cmd += ["--feat_conds", feat]
        safe(f"scan-{tag}", lambda c=cmd: run(c, CFM, "scan"))

    safe("tiger", lambda: run([
        PY, TIGER / "tools/tiger_zeroshot.py", "--num", "4", "--steps", "30",
        "--out", TIGER / "results/tiger_zeroshot"], TIGER, "tiger"))

    log("POST QUEUE DONE")
    status("all", "done")


if __name__ == "__main__":
    main()
