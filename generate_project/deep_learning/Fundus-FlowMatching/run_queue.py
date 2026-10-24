#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""无人值守实验队列（作者 2026-10-23 批准）

顺序：
  0) 等 cfm256_lesion 训练结束 → 权重归位 CFM/models/
  1) 评估病灶加权模型（60 张 + B 域指标）
  2) ROADMAP-1：轨迹对齐直方图匹配（8 张真实 severe 投影）
  3) ROADMAP-7：Tiger 零样本（normal/severe 各 4 张）
  4) ROADMAP-2：cfm128_align 训练（320 轮）
  5) 评估 cfm128_align（60 张 + 指标）
  6) ROADMAP-8：cfm128_text 训练（320 轮）
  7) 评估 cfm128_text（60 张 + 指标）

每步写日志与 RUN_QUEUE_STATUS.json；单步失败不阻塞后续。
"""
import json
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
TIGER = REPO / "generate_project/pretrained/TextGuidedDiffusion"
PY = sys.executable
LOG = CFM.parent / "RUN_QUEUE.log"
STATUS = CFM.parent / "RUN_QUEUE_STATUS.json"


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(step, state, extra=None):
    data = {"step": step, "state": state, "time": datetime.now().isoformat()}
    if extra:
        data.update(extra)
    STATUS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def gpu_mem():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                     "--format=csv,noheader,nounits"], timeout=5, text=True)
        return int(o.strip().splitlines()[0])
    except Exception:
        return 0


def wait_gpu_free(limit=1800, need=3):
    ok = 0
    t0 = time.time()
    while time.time() - t0 < limit:
        if gpu_mem() < 1500:
            ok += 1
            if ok >= need:
                return True
        else:
            ok = 0
        time.sleep(10)
    return False


def wait_temp_ok(limit=70):
    for _ in range(30):
        try:
            t = int(subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu",
                                             "--format=csv,noheader"], timeout=5, text=True).strip())
        except Exception:
            return
        if t <= limit:
            return
        log(f"  [temp] {t}C > {limit}C, wait 60s")
        time.sleep(60)


def run(cmd, cwd, name, stdout_file=None):
    log(f"  RUN {name}: {' '.join(str(c) for c in cmd)}")
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


def wait_lesion_final(timeout=8 * 3600):
    final = CFM / "results/cfm256_lesion/models/final_model.pth"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if final.exists() and wait_gpu_free(limit=180):
            return final
        time.sleep(30)
    cks = sorted((CFM / "results/cfm256_lesion/models").glob("checkpoint_epoch_*.pth"))
    return cks[-1] if cks else None


def eval_model(ckpt, tag, cond_path, img_size, epochs_note=""):
    fake = REPO / "eval_data" / tag
    if not (fake / "singles").exists() or len(list((fake / "singles").glob("*.png"))) < 60:
        run([PY, DDPM / "generate.py", "--checkpoint", ckpt, "--num_images", "60",
             "--img_size", img_size, "--base_dim", "128", "--seed", "100",
             "--sampling_steps", "50", "--output_dir", fake, "--grid_size", "0",
             "--cond_path", cond_path], DDPM, f"gen-{tag}")
    fake_dir = fake / "singles"
    run([PY, REPO / "eval/metrics_common.py", "--real", REPO / "eval_data/real_severe_train",
         "--fake", fake_dir, "--img_size", "256", "--device", "cuda", "--json"],
        REPO, f"metrics-common-{tag}", REPO / f"eval_data/{tag}_metrics.json")
    env_note = "PYTHONIOENCODING=utf-8"
    cmd = [PY, REPO / "eval/metrics_fundus.py", "--real", REPO / "eval_data/real_severe_train",
           "--fake", fake_dir, "--model", tag, "--img_size", "256", "--device", "cuda",
           "--cond_path", CFM / "data/conditions_combined_256", "--json"]
    log(f"  RUN metrics-fundus-{tag} ({env_note})")
    import os
    env = dict(os.environ); env["PYTHONIOENCODING"] = "utf-8"
    with open(REPO / f"eval_data/{tag}_fundus_metrics.json", "w", encoding="utf-8") as fo:
        r = subprocess.run([str(c) for c in cmd], cwd=str(REPO), stdout=fo,
                           stderr=subprocess.DEVNULL, env=env)
    log(f"  END metrics-fundus-{tag}: exit={r.returncode}")


def safe(name, fn):
    status(name, "running")
    try:
        fn()
        status(name, "done")
    except Exception:
        log("STEP " + name + " ERROR\n" + traceback.format_exc())
        status(name, "error")


def step_lesion_eval():
    dst = CFM / "models/cfm256_lesion_final_model.pth"
    if not dst.exists():
        src = CFM / "results/cfm256_lesion/models/final_model.pth"
        if src.exists():
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
    eval_model(dst, "cfm256_lesion", CFM / "data/conditions_combined_256", 256)


def step_tta():
    run([PY, CFM / "tools/tta_hist_match.py",
         "--ckpt", CFM / "models/cfm256_final_model.pth",
         "--src", REPO / "fundus/HRSG_4_758_sq256/train/severe",
         "--cond", CFM / "data/conditions_combined_256",
         "--num_ref", "12", "--s_target", "0.5", "--steps", "100",
         "--limit", "8", "--out", CFM / "results/tta_hm"], DDPM, "tta_hm")


def step_tiger():
    run([PY, TIGER / "tools/tiger_zeroshot.py", "--num", "4", "--steps", "30",
         "--out", TIGER / "results/tiger_zeroshot"], TIGER, "tiger_zeroshot")


def step_align_train():
    run([PY, CFM / "train/train_cfm.py", "--preset", "cfm128_align"], CFM, "cfm128_align_train")


def step_align_eval():
    eval_model(CFM / "results/cfm128_align/models/final_model.pth", "cfm128_align",
               CFM / "data/conditions_combined_128", 128)


def step_text_train():
    run([PY, CFM / "train/train_cfm.py", "--preset", "cfm128_text"], CFM, "cfm128_text_train")


def step_text_eval():
    eval_model(CFM / "results/cfm128_text/models/final_model.pth", "cfm128_text",
               CFM / "data/conditions_combined_128", 128)


def main():
    LOG.write_text("", encoding="utf-8")
    log("QUEUE RESTART")
    for name, fn in [
        ("eval_lesion", step_lesion_eval),
        ("tta_hm", step_tta),
        ("tiger", step_tiger),
        ("cfm128_align_train", step_align_train),
        ("cfm128_align_eval", step_align_eval),
        ("cfm128_text_train", step_text_train),
        ("cfm128_text_eval", step_text_eval),
    ]:
        safe(name, fn)
    log("QUEUE DONE")
    status("all", "done")


if __name__ == "__main__":
    main()
