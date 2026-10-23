# models/ —— CFM 族模型权重

命名：`<实验名>_final_model.pth`，例如：

- `cfm128_final_model.pth`：CFM@128（320 轮）
- `cfm256_final_model.pth`：CFM@256（320 轮，作者 97/100）
- `cfm256_lesion_final_model.pth`：病灶加权续训（跑完后从 results/cfm256_lesion/models 移入）

权重按根 `.gitignore` 不入 git；本 README 只用于保留目录与命名约定。
