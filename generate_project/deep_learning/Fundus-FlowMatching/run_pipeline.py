#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""眼底侧无人值守流水线（OCT 复现暂停后的新序列）

顺序：
  0) 等 cfm128_featC 训练结束 → 权重归位 CFM/models/
  1) 评估 featC（60 张 + 多样性 + B 域指标）
  2) 轨迹对齐 HM（CFM@256，真实 severe 8 张投影）
  3) cfm128_align 训练 + 评估
  4) cfm128_text 训练 + 评估
  5) cfm128_notime 训练 + 评估
  6) cfm128_featA 训练 + 评估

每步写 RUN_PIPELINE.log / RUN_PIPELINE_STATUS.json；单步失败不阻塞后续。
"""
import json
import os
import shutil
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
PY = sys.executable
LOG = CFM.parent / "RUN_PIPELINE.log"
STATUS = CFM.parent / "RUN_PIPELINE_STATUS.json"


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
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


def wait_gpu_free(limit=1800, need=3):
    ok, t0 = 0, time.time()
    while time.time() - t0 < limit:
        ok = ok + 1 if gpu_mem() < 1500 else 0
        if ok >= need:
            return True
        time.sleep(10)
    return False


def wait_temp_ok(limit=70):
    for _ in range(30):
        try:
            t = int(subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu",
                                             "--format=csv,noheader"], timeout=5, text=True))
        except Exception:
            return
        if t <= limit:
            return
        log(f"  [temp] {t}C, wait 60s")
        time.sleep(60)


def run(cmd, cwd, name, stdout_file=None):
    log(f"  RUN {name}")
    wait_temp_ok()
    if stdout_file:
        with open(stdout_file, "w", encoding="utf-8") as fo:
            r = subprocess.run([str(c) for c in cmd], cwd=str(cwd), stdout=fo,
                               stderr=subprocess.DEVNULL)
    else:
        with open(LOG, "a", encoding="utf-8") as fo:
            r = subprocess.run([str(c) for c in cmd], cwd=str(cwd), stdout=fo,
                               stderr=subprocess.STDOUT)
    log(f"  END {name}: exit={r.returncode}")
    return r.returncode


def wait_file(path, timeout=8 * 3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if Path(path).exists() and wait_gpu_free(limit=180):
            return True
        time.sleep(30)
    return False


def eval_generic(ckpt, tag, cond_path, img_size):
    fake = REPO / "eval_data" / tag
    if len(list((fake / "singles").glob("*.png"))) < 60:
        run([PY, DDPM / "generate.py", "--checkpoint", ckpt, "--num_images", "60",
             "--img_size", img_size, "--base_dim", "128", "--seed", "100",
             "--sampling_steps", "50", "--output_dir", fake, "--grid_size", "0",
             "--cond_path", cond_path], DDPM, f"gen-{tag}")
    fake_dir = fake / "singles"
    run([PY, REPO / "eval/metrics_common.py", "--real", REPO / "eval_data/real_severe_train",
         "--fake", fake_dir, "--img_size", "256", "--device", "cuda", "--json"],
        REPO, f"metrics-common-{tag}", REPO / f"eval_data/{tag}_metrics.json")
    env = dict(os.environ); env["PYTHONIOENCODING"] = "utf-8"
    with open(REPO / f"eval_data/{tag}_fundus_metrics.json", "w", encoding="utf-8") as fo:
        r = subprocess.run([str(c) for c in [
            PY, REPO / "eval/metrics_fundus.py", "--real", REPO / "eval_data/real_severe_train",
            "--fake", fake_dir, "--model", tag, "--img_size", "256", "--device", "cuda",
            "--cond_path", CFM / "data/conditions_combined_256", "--json"]],
            cwd=str(REPO), stdout=fo, stderr=subprocess.DEVNULL, env=env)
    log(f"  END metrics-fundus-{tag}: exit={r.returncode}")


def step_featC():
    dst = CFM / "models/cfm128_featC_final_model.pth"
    src = CFM / "results/cfm128_featC/models/final_model.pth"
    if not dst.exists() and src.exists():
        shutil.move(str(src), str(dst))
    run([PY, CFM / "tools/eval_feat_model.py", "--ckpt", dst,
         "--conds", CFM / "data/handcrafted/conds_severe.npy",
         "--tag", "cfm128_featC", "--num", "60",
         "--cond", CFM / "data/conditions_combined_128"], CFM, "eval-featC")


def step_hm():
    run([PY, CFM / "tools/tta_hist_match.py",
         "--ckpt", CFM / "models/cfm256_final_model.pth",
         "--src", REPO / "fundus/HRSG_4_758_sq256/train/severe",
         "--cond", CFM / "data/conditions_combined_256",
         "--num_ref", "12", "--s_target", "0.5", "--steps", "100",
         "--limit", "8", "--out", CFM / "results/tta_hm"], DDPM, "tta_hm")


def train(preset):
    run([PY, CFM / "train/train_cfm.py", "--preset", preset], CFM, f"train-{preset}")


def main():
    LOG.write_text("", encoding="utf-8")
    log("PIPELINE START")
    steps = []
    steps.append(("featC", lambda: (wait_file(CFM / "results/cfm128_featC/models/final_model.pth"),
                                    step_featC())))
    steps.append(("hm", step_hm))
    steps.append(("align", lambda: (train("cfm128_align"),
                                    eval_generic(CFM / "results/cfm128_align/models/final_model.pth",
                                                 "cfm128_align", CFM / "data/conditions_combined_128", 128))))
    steps.append(("text", lambda: (train("cfm128_text"),
                                   eval_generic(CFM / "results/cfm128_text/models/final_model.pth",
                                                "cfm128_text", CFM / "data/conditions_combined_128", 128))))
    steps.append(("notime", lambda: (train("cfm128_notime"),
                                     eval_generic(CFM / "results/cfm128_notime/models/final_model.pth",
                                                  "cfm128_notime", CFM / "data/conditions_combined_128", 128))))
    steps.append(("featA", lambda: (train("cfm128_featA"),
                                    run([PY, CFM / "tools/eval_feat_model.py",
                                         "--ckpt", CFM / "results/cfm128_featA/models/final_model.pth",
                                         "--conds", CFM / "data/retfound_cfp/conds_severe.npy",
                                         "--tag", "cfm128_featA", "--num", "60",
                                         "--cond", CFM / "data/conditions_combined_128"],
                                        CFM, "eval-featA"))))
    for name, fn in steps:
        status(name, "running")
        try:
            fn()
            status(name, "done")
        except Exception:
            log(f"STEP {name} ERROR\n" + traceback.format_exc())
            status(name, "error")
    log("PIPELINE DONE")
    status("all", "done")


if __name__ == "__main__":
    main()
