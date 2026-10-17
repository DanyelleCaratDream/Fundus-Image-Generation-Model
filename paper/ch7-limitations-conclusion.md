# 7. Limitations & Conclusion — stub

> 状态：stub。素材：docs/11 第五节"过度声称高危区"逐条自查 + docs/14 口径表 #15。**首批写作不写本章。**

## 7.1 Limitations（去虚清单）

- 单人人工分（N=6 校准集）——尺度局限（口径 #2）
- 干净样本率 ~25-30%（DL-EX-014 常态），展示图为手工挑选（纪律 #6）
- 老师官方 split 跨 split 泄漏（113 对相似）→ 基线 QWK 0.747@256 含乐观偏差，作为 dataset limitation 记录（口径 #15）
- 128px 生成 → SR 到 256 交付（非原生 256）
- 六维门控总分为 min-max 相对分，模型库扩大会整体平移（口径 #2）
- 多 seed / 统计显著性未做（路线 B）

## 7.2 Conclusion

（待写——回收 C1–C4；结尾接 Tiger-Model 式开源路线展望）
