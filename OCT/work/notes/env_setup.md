# 环境搭建记录与偏离说明

> 环境：AutoDL GPU 容器，NVIDIA GeForce RTX 4090（24 GB）、驱动 580.105.08、CUDA 13.0
> 持久工作目录：`D:/AI_Model_Project_for_Fundus_Color_Images/OCT/work`
> 记录日期：2026-10-24

本文件记录环境搭建过程中**实际遇到的问题与被迫做出的偏离**，供复现报告引用。
凡与论文/仓库设定不一致之处，均在此显式列出，不在正文中隐去。

---

## 1. 磁盘布局与"环境必须放在持久盘"的原因

实测磁盘布局（`df -h`）：

| 挂载点 | 容量 | 说明 |
|---|---|---|
| `/`（overlay） | **30 GB** | 系统盘。Python 发行版、pip 缓存、`/tmp` 都在这里 |
| `/root/autodl-tmp`（`/dev/md0`） | **50 GB** | **唯一会被平台自动同步并保留的目录** |

平台约定：只有 `/root/autodl-tmp` 在 GPU 停止/释放后仍然保留，其余目录会被丢弃。

**遇到过的失败**：首次安装依赖时使用系统 Python（`/usr/local/bin/python3.12`，位于 30 GB overlay），
在下载 `nvidia-cusparse-cu12` 时以 `[Errno 28] No space left on device` 失败——
当时 overlay 仅剩 1.9 GB，而清理 `/tmp/pip-*` 与 `/root/.cache/pip` 后也只恢复到 11 GB。

**对策**：把虚拟环境整体建在持久盘上，并让所有临时文件也落在持久盘：

```bash
python3.12 -m venv D:/AI_Model_Project_for_Fundus_Color_Images/OCT/work/venv
export TMPDIR=D:/AI_Model_Project_for_Fundus_Color_Images/OCT/work/tmp
export PIP_CACHE_DIR=D:/AI_Model_Project_for_Fundus_Color_Images/OCT/work/pip-cache
```

好处有二：既绕开系统盘容量限制，又让环境在 GPU 重启后仍然存在，无需重装。

---

## 2. 依赖源的实测速度与选择

| 源 | 实测速度 | 结论 |
|---|---|---|
| `download-r2.pytorch.org`（官方 CUDA wheel 源） | **119 KB/s** | 841 MB 的 torch 需约 2 小时，不可用 |
| 清华 PyPI 镜像 | 15–18 MB/s | 采用（普通包） |
| 阿里云 pytorch-wheels | 10.2 MB/s | 文件本身可达，但**不是 PEP 503 索引**，pip 无法据此解析版本 |
| 上海交大 pytorch-wheels | 9.5 MB/s | 同上 |

---

## 3. 与仓库 `requirements.txt` 的偏离（必须记录）

仓库固定了 `torch==2.10.0+cu126` 与 `torchvision==0.25.0+cu126`。
`+cu126` 是 PyTorch 官方的**本地版本标签**，只存在于 `download.pytorch.org` 的 wheel 索引中；
该源在本机不可用（见上表），而国内镜像虽然托管了同名文件，却**未提供 pip 可解析的索引页**。

实测报错：

```
ERROR: Could not find a version that satisfies the requirement torch==2.10.0+cu126
ERROR: No matching distribution found for torch==2.10.0+cu126
```

**采用的方案**：从 PyPI（清华镜像）安装 **`torch==2.10.0` / `torchvision==0.25.0`**，
即与仓库 pin 的**同一版本号**，区别仅在于 CUDA 运行时由 `cu126` 变为 PyPI 默认捆绑的 **`cu12.8`**
（`nvidia-*-cu12 12.8.x`）。

| 项目 | 仓库 pin | 本环境 | 影响评估 |
|---|---|---|---|
| torch | 2.10.0+cu126 | **2.10.0**（cu12.8 捆绑） | 版本号一致；CUDA 运行时 12.6→12.8，对 RTX 4090（sm_89）无功能差异 |
| torchvision | 0.25.0+cu126 | **0.25.0** | 同上 |
| nvidia-* | 12.6.x 系列 | 12.8.x 系列 | 由 torch 自身元数据决定，未手工覆盖 |

**为什么要加约束文件**：过滤掉 `torch` 行后，`pytorch-lightning==2.6.1` 等包的
`torch>=2.1.0` 约束会让 pip 在回溯时把 torch 升级到最新（曾观察到升到 `2.14.1+cu130`），
进而整体切到 CUDA 13 依赖栈，与仓库 pin 偏离更远。
因此第二轮安装使用 `-c logs/constraints.txt`（内容为 `torch==2.10.0` 与 `torchvision==0.25.0`）锁定，
并**从零重建 venv** 以清除第一轮误装的 torch 2.14。

> 记录原则：这是"镜像可得性"导致的被迫偏离，不是有意改设定。若后续能接入官方 CUDA wheel 源，
> 应回到 `+cu126` 并重跑关键实验。

---

## 4. 代码同步通道的替代方案

平台提供的文件传输（`gpu_upload_file`）经 OSS 完成，其端点为 `https://wisdisk.qiewen.com`。
**该端点在 GPU 节点上不可达**（`curl` 直接超时），`rclone` 传输进程会永久挂起且不报错——
实测有 3 个 rclone 进程分别挂了 10~20 分钟仍未完成。

**替代通道**（实测可用、带完整性校验）：

1. 本地打包 `repro/code` → `tar.gz` → base64 → `curl --data-binary @- https://paste.rs`
2. GPU 侧 `curl <返回的 URL> | base64 -d > bundle.tar.gz` → 比对 md5 → `tar xzf`

已验证往返 md5 一致。该流程封装为 `repro/_transfer/push.sh`。
注意日志中出现的 `/tmp/rclone.conf` 内含明文 S3 凭据，本笔记已脱敏，不记录凭据内容。

---

## 5. 需留意的资源约束（对应风险 R6）

| 资源 | 论文设定 | 本环境 | 应对 |
|---|---|---|---|
| 显存 | RTX A6000 **48 GB** | RTX 4090 **24 GB** | 已设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；若 224×224 + batch 16 OOM，则降 batch 并记录为配置偏离 |
| 训练时长 | 300,000 次迭代约 **80 h** | 按用户选定的 quick 档（约 3 万次迭代）先验证链路 | full 档需先评估 GPU 会话时长预算 |
| 持久盘 | — | 50 GB | 数据 + 检查点 + 推理输出需控制在 50 GB 内；推理输出每卷约 25 MB（128×224×224×4B），60 卷约 1.5 GB |

---

## 6. 软件版本核对清单（安装完成后填写实测值）

见 `logs/env_versions.txt`（由 `scripts/check_env.sh` 生成），此处不预填，避免写入未经实测的版本号。