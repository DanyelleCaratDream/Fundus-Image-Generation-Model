# [旧实现——有缺陷，仅用于对照] 分母只计单通道 → 加权项为预期 3×；λ=0 也不等于 F.l1_loss。
# 修复版见 train_loss_block_fixed.py
# train.py 第 1239-1268 行（加权损失实现片段）

                if is_cfm:
                    t_cont = torch.rand(batch, device=device)
                    t = diffusion.t_cont_to_index(
                        t_cont, align=bool(getattr(args, "cfm_time_align", False)))
                    x_t, v_target, _noise = diffusion.cfm_path(real_data, t_cont)
                    predicted = model(x_t, t, cond=cond, text_emb=text_emb_b)
                    if args.loss_type == "mse":
                        loss = F.mse_loss(predicted, v_target)
                    else:
                        loss = F.l1_loss(predicted, v_target)
                    # 病灶区域加权（可选）：w=1+λ·mask，按权重归一化保持 loss 尺度
                    if getattr(args, "lesion_weight_dir", None) and len(batch_data) > 3:
                        lw = batch_data[2].to(device, dtype=predicted.dtype)
                        lw = 1.0 + float(getattr(args, "lesion_lambda", 5.0)) * lw
                        loss = (torch.abs(predicted - v_target) * lw).sum() / lw.sum()
                    # CFM：t→1 靠近数据，对应 DDPM 的低噪声段
                    low_noise_mask = t_cont > (1.0 - float(lpips_t) / args.timesteps)
                    if low_noise_mask.any():
                        x1_pred = diffusion.cfm_predict_x1(
                            x_t[low_noise_mask], t_cont[low_noise_mask], predicted[low_noise_mask]
                        ).clamp(-1.0, 1.0)
                        pred_mean = x1_pred.mean(dim=[0, 2, 3])
                        loss = loss + args.color_weight * F.mse_loss(pred_mean, target_mean)
                        if lpips_fn is not None:
                            lpips_val = lpips_fn(x1_pred, real_data[low_noise_mask], normalize=True).mean()
                            loss = loss + args.lpips_weight * lpips_val
                else:
                    t = torch.randint(0, args.timesteps, (batch,), device=device, dtype=torch.long)
                    noisy_images, noise = diffusion.q_sample(real_data, t)