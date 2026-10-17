# Continue Guide - 2026-10-16（v24 — Phase D 进行中：数据卫生三连查完成 + D1/D2 预登记待跑 + 文档体系论文向重整）

> **⚠️ 反失步纪律（2026-10-16 起）**：本文件只记「**当前位置 + 下一步 + 既定纪律**」。
> **所有实验数字一律以 `generate_project/deep_learning/EXPERIMENTS.md`（DL-EX-xxx）和 `generate_project/machine_learning/EXPERIMENTS.md`（EX-xxx）为唯一来源**，本文件不复制数字。每轮工作结束更新本文件的日期、版本与「当前位置」。
> **当前任务（2026-10-17 中午）**：① 文档体系论文向重整 ✅；② **D1 ❌ 完成**（193×256×bd64×500轮，0/8 无结构；归因修正：与 film_l1lpips 相比容量+分辨率同时变动，数据量假说**未**被裁决——数字见 EXPERIMENTS.md DL-EX-019）；③ **D2 ⏳ 转云算力**（作者定夺：本地显存溢出弃跑、否决 batch1；D2=矩阵唯一缺格 193×256×bd128，云端 batch≥2 跑 780-1000 轮）；④ 待办：为作者准备云端跑配置清单。

---

## 一、当前位置：Phase D —— 在老师数据集上做 severe 生成

**总目标（= 项目最终目的）：产出一篇优秀的 paper。** 论文第一、实用性第二（2026-10-16 用户明确）。
论文路线：**A《小样本眼底生成：失效机理与评估方法》先行 → B（+Phase D 下游验证 TSTR）叠加**。
论文写作宪法：`docs/14-Paper-Spine.md`（主张→证据映射 + 口径统一表 + 术语中英对照）。数据权威描述：`docs/15-Data-Documentation.md`。

**Phase D 链路**：老师数据集 `fundus/HRSG_4_758_v0.3.4`（758 张 = train 552/val 135/test 71，四分类）
→ 生成 severe 新图（train 侧数据训练，不碰 val/test）→ 老师审核 → **替换 train 里 76 张数据集自带的垂直翻转 `_aug_1` 副本**
→ 扩充后数据集训分类器（TSTR，对比基线 QWK 0.747@256）。

**实验链条**（数字见 EXPERIMENTS.md）：
- DL-EX-015（76张@256 从零）❌ / 016（76张@128）❌：loss 正常收敛但生成全糊 → 主假说「数据量不足」
- DL-EX-017（容量对照）⛔ 搁置（设计与 015 双变量不干净）
- DL-EX-018（微调 A 域模型）⚠️ 部分成功：干净样本 ~25% 与 014 常态持平，但失败样本崩得更狠、128px 与 Phase D 不符
- **DL-EX-019（D1：combined 193张@256 bd64 500轮）/ 020（D2：bd128 300轮）⏳ 预登记完毕，待启动**
  - 判定（预注册）：**不看 loss 只看 `--use_raw` 出图**；多数样本可辨认眼底结构 = 成功（对比 015 的 0/8）；300 轮仍无结构即可判失败
  - D1 与 015 只差数据量（76→193）= 干净单变量对照；D2 加容量做二次检验
- 数据：`fundus/combined_sq256` 193 张 + `conditions_combined` 193 mask（一一对应）

## 二、下一步（依序）

1. ~~启动 D1/D2~~（🏃 2026-10-17 00:25 运行中，独立进程，会话关闭不影响）
2. **训练完成后**：`nvidia-smi -rgc` 无需执行（时钟未锁成功）；检查保险丝旗标 `D:/tmp/D1D2_FUSE_TRIPPED.txt` 是否存在（存在 = 曾失控断电，看 D1.log 判进度）；出图判定 → 填 EXPERIMENTS.md DL-EX-019/020 → **定正式生成路线**（combined 模型 vs B 微调 vs 预训练+LoRA [报告 07]）
3. 正式生成替换图（256px，300 DPI 论文规格同步出图，手工挑图并注明筛选事实）→ 老师审核 → 替换 76 张
4. 专家盲评（docs/13 协议；评分表模板未做）
5. TSTR：扩充数据集训分类器 vs 基线 QWK 0.747@256
6. 论文路线 A 动笔（docs/14 脊柱 → 逐节素材指针）

## 三、数据卫生（2026-10-16 三连查，全部肉眼复核）

详见 **`docs/15-Data-Documentation.md`**。一句话版：
- 旋转副本 **0**（旧派生集才有几何增强，已移出磁盘；v1 误报 74 对已修正）
- 池内重复：网络 163→153 唯一；老师 severe 76→**67** 唯一；跨池同图 16 对
- **跨 split 泄漏**：老师官方 split train↔val/test 有像素级同图（severe 3 对 MSE=0 + moderate 16794≡16788/16796 + 网络图撞 val/test 5 对）；全树 113 对相似 → 数据集按图非按眼划分
- 处置：combined_sq256 两批清理 239→**193**，移出文件进 `fundus/_removed_dups_sq256/`（MANIFEST 可回滚）；**官方数据集未动**
- ✅ 299 vs 330 已确认（2026-10-16 作者）：差的 31 张是作者手动**质量淘汰**的；且 299 中仍有作者认为不过关的图，作者认可的"过关"集合 = `all_ORIGINAL_sq256`（163）——质量层级与论文影响见 docs/15 §四

## 四、📣 转告老师两件事（长期挂起直至传达）

1. 官方 split 存在跨 split 同图 → 分类器基线 QWK 0.747@256 含乐观偏差（清单在 docs/15 §2.3）
2. 建议按眼/序列分组重划分（老师决定；我们不擅自改官方数据）

## 五、既定纪律（standing rules，持续有效）

- **必须看图，不能只看统计量**（DL-EX-015 排查教训：统计量正常但图是糊块）
- 出图判定用 `generate.py --use_raw`（EMA 在续训 checkpoint 上可能退化）
- **六段式汇报流程（强制）**：是什么 → 怎么做（略写）→ 为什么 → 结果（图/脚本分/人眼分）→ 致命缺点 → 下一步；每实验做完先问用户有无补充，确认后才进行下一步
- 评估新方法统一 `--num_images 60`；**C2ST 必跑**不得 `--skip_c2st`（口径不均教训）；复制检测必配
- Phase D：**test 保持全真实**；生成器训练数据不含 val/test；分辨率 256×256（2026-10-16 定，取代 512 旧决策）
- 人工分 vs 六维分两套口径**永不混排**（docs/14 口径表 #2）
- 笔记本 GPU：`--cool_sleep -0.5`；长任务前 `nvidia-smi` 看温度；数据先预缩到目标分辨率（否则慢 15 倍）
- 归档惯例：**先归档不销毁**，进 `_archive/` 并在 ARCHIVE.md 登记
- 双远端：`git push github main && git push gitee main`

## 六、快速命令

```bash
# 实验链进度（Windows 下双击 查看实验进度.bat 亦可）
cd generate_project/deep_learning/Fundus-Diffusion/ddpm && python check_progress.py

# 启动 D1/D2（待用户下令）
cd generate_project/deep_learning/Fundus-Diffusion/ddpm && bash run_D1_D2.sh

# 评估 + 评分 + 出图（新方法：JSON 丢进 eval_data/ 后）
python eval/score_scheme.py --scorecard && python eval/plot_metrics.py

# 推送双远端
git push github main && git push gitee main
```

## 七、文档体系（2026-10-16 重整后）

- 入口：根 `README.md`（论文向门面）→ `docs/00-Doc-Index.md`（全图）
- 论文脊柱：`docs/14-Paper-Spine.md` ｜ 数据权威：`docs/15-Data-Documentation.md` ｜ 就绪度：`docs/11` ｜ 文献定位：`docs/12` ｜ 盲评协议：`docs/13`
- 已归档：docs/03/04/05（内容被 REPORT + Work-Guide 覆盖，路径已失效）＋ `稿子.md`（参考模板）——见 `_archive/ARCHIVE.md`
