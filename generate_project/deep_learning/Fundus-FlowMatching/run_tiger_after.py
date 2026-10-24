#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""等眼底流水线结束后，自动跑 Tiger-Model 零样本观察（补进队列的最后一项）。

监视 RUN_PIPELINE_STATUS.json；当 step=all 且 state in {done,error} 时启动 Tiger。
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(r"D:/AI_Model_Project_for_Fundus_Color_Images")
FLOW = REPO / "generate_project/deep_learning/Fundus-FlowMatching"
TIGER = REPO / "generate_project/pretrained/TextGuidedDiffusion"
STATUS = FLOW / "RUN_PIPELINE_STATUS.json"
OUT_STATUS = FLOW / "TIGER_STATUS.json"
LOG = FLOW / "RUN_TIGER.log"
PY = sys.executable


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    LOG.write_text("", encoding="utf-8")
    log("TIGER WATCHER START")
    while True:
        try:
            d = json.loads(STATUS.read_text(encoding="utf-8"))
        except Exception:
            d = {}
        if d.get("step") == "all" and d.get("state") in ("done", "error"):
            break
        time.sleep(60)

    log(f"pipeline finished: {d}")
    OUT_STATUS.write_text(json.dumps({"state": "running", "time": datetime.now().isoformat()},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
    cmd = [PY, TIGER / "tools/tiger_zeroshot.py", "--num", "4", "--steps", "30",
           "--out", TIGER / "results/tiger_zeroshot"]
    with open(LOG, "a", encoding="utf-8") as fo:
        r = subprocess.run([str(c) for c in cmd], cwd=str(TIGER), stdout=fo,
                           stderr=subprocess.STDOUT)
    log(f"tiger exit={r.returncode}")
    OUT_STATUS.write_text(json.dumps({"state": "done" if r.returncode == 0 else "error",
                                      "exit": r.returncode,
                                      "time": datetime.now().isoformat()},
                                     ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
