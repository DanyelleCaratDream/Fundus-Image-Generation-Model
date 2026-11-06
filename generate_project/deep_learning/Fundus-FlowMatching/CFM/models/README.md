# models/ —— CFM 族模型权重

命名：`<实验名>_final_model.pth`，例如：

- `cfm128_final_model.pth`：CFM@128（320 轮）
- `cfm256_final_model.pth`：CFM@256（320 轮，作者 97/100）
- `cfm256_lesion_final_model.pth`：渗出加权续训（λ=8，已完成；仅渗出，出血未加权）

权重按根 `.gitignore` 不入 git；本 README 只用于保留目录与命名约定。
