#!/usr/bin/env python3
"""生成 TTA-Flow 仓库所需的 CSV 数据划分文件。

官方仓库通过 `data/dataset_configs/<name>.csv` 定义数据划分，格式为两列：

    volume,mask
    <volume.npy 路径>,<mask.npy 路径>

本脚本扫描转换后的 .npy 目录，为三个设备各生成一个 CSV。相对路径以
`--prefix` 指定（默认 `/app/data/retouch`，即仓库 Dockerfile 内的挂载点）。

用法
----
    python create_dataset_configs.py --npy-root /path/to/retouch_npy \
        --out-dir /path/to/tta-flow/data/dataset_configs \
        --prefix /path/to/retouch_npy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
log = logging.getLogger("create-configs")

DEVICES = ["spectralis", "cirrus", "topcon"]
# 论文报告的体积数，用于交叉校验
EXPECTED = {"spectralis": 24, "cirrus": 24, "topcon": 22}


def build_rows(npy_root: Path, device: str, prefix: Path) -> list[dict[str, str]]:
    """为单个设备收集 (volume, mask) 行。缺少标注的体积会被记录但不写入。"""
    vol_dir = npy_root / device
    ann_dir = npy_root / "annotations" / device
    if not vol_dir.exists():
        log.warning("设备目录不存在，跳过: %s", vol_dir)
        return []

    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for vol in sorted(vol_dir.glob("*.npy")):
        ann = ann_dir / vol.name
        if not ann.exists():
            missing.append(vol.name)
            continue
        rows.append(
            {
                "volume": str(prefix / device / vol.name),
                "mask": str(prefix / "annotations" / device / ann.name),
            }
        )
    if missing:
        log.warning("%s: %d 个体积缺少标注，已从 CSV 中排除: %s",
                    device, len(missing), missing[:5])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 TTA-Flow 数据划分 CSV")
    parser.add_argument("--npy-root", required=True, type=Path, help="转换后的 .npy 根目录")
    parser.add_argument("--out-dir", required=True, type=Path, help="CSV 输出目录")
    parser.add_argument("--prefix", type=Path, default=None,
                        help="写入 CSV 的路径前缀（默认与 --npy-root 相同）")
    parser.add_argument("--hydra-dir", type=Path, default=None,
                        help="同时生成 Hydra 数据配置目录，例如 "
                             "<repo>/preferences/data。会写入 retouch_<device>_local.yaml，"
                             "其 data_dir 指向本脚本生成的 CSV 绝对路径。")
    args = parser.parse_args()

    if not args.npy_root.exists():
        log.error("--npy-root 不存在: %s", args.npy_root)
        return 2

    prefix = args.prefix or args.npy_root
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ok = True
    for device in DEVICES:
        rows = build_rows(args.npy_root, device, prefix)
        if not rows:
            log.error("%s: 无可用样本", device)
            ok = False
            continue

        name = f"retouch_{device}"
        out_path = args.out_dir / f"{name}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)

        exp = EXPECTED[device]
        flag = "✅" if len(rows) == exp else "⚠️"
        if len(rows) != exp:
            ok = False
        log.info("%s %s: %d 行（期望 %d） -> %s", flag, device, len(rows), exp, out_path)

    if args.hydra_dir is not None:
        args.hydra_dir.mkdir(parents=True, exist_ok=True)
        for device in DEVICES:
            cfg_path = args.hydra_dir / f"retouch_{device}_local.yaml"
            cfg_path.write_text(
                f'name: "RETOUCH_{device}"\n'
                f'data_dir: "{(args.out_dir / f"retouch_{device}.csv").resolve()}"\n',
                encoding="utf-8",
            )
        log.info("已生成 %d 个 Hydra 数据配置 -> %s", len(DEVICES), args.hydra_dir)

    log.info("注意：本脚本生成的是 *_local.csv 之外的划分文件；"
             "仓库自带的 retouch_original_*.csv 使用 /app/data/retouch 前缀（Docker 内路径），"
             "若在裸机运行请使用此处生成的 CSV 或相应覆盖 data.data_dir。")
    log.info("结果: %s", "✅ 通过" if ok else "❌ 数量与论文不一致，请核对数据完整性")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
