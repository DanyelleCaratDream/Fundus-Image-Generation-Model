# train.py CFM 损失块（2026-11-06 修复后，含上下文）
# 修复：w=(1+λ·lw).expand_as(predicted)；分子/分母同形状；λ=0 严格等于 F.l1_loss

 1255|                         loss = F.mse_loss(predicted, v_target)
 1256|                     else:
 1257|                         loss = F.l1_loss(predicted, v_target)
 1258|                     # 渗出区域加权（可选）：w=1+λ·mask，分子/分母同形状归一化。
 1259|                     # 2026-11-06 修复：lw 为 [B,1,H,W] 而 predicted 为 [B,3,H,W]，
 1260|                     # 旧写法分母只算 1 个通道 → 加权项恒为预期加权均值的 3 倍；
 1261|                     # λ=0 时也不等于 F.l1_loss。expand_as 后两者形状一致。
 1262|                     if getattr(args, "lesion_weight_dir", None) and len(batch_data) > 3:
 1263|                         lw = batch_data[2].to(device, dtype=predicted.dtype)
 1264|                         w = (1.0 + float(getattr(args, "lesion_lambda", 5.0)) * lw).expand_as(predicted)
 1265|                         if args.loss_type == "mse":
 1266|                             err = (predicted - v_target) ** 2
 1267|                         else:
 1268|                             err = (predicted - v_target).abs()
 1269|                         loss = (err * w).sum() / w.sum()
 1270|                     # CFM：t→1 靠近数据，对应 DDPM 的低噪声段
 1271|                     low_noise_mask = t_cont > (1.0 - float(lpips_t) / args.timesteps)
 1272|                     if low_noise_mask.any():
 1273|                         x1_pred = diffusion.cfm_predict_x1(
 1274|                             x_t[low_noise_mask], t_cont[low_noise_mask], predicted[low_noise_mask]
 1275|                         ).clamp(-1.0, 1.0)
 1276|                         pred_mean = x1_pred.mean(dim=[0, 2, 3])
 1277|                         loss = loss + args.color_weight * F.mse_loss(pred_mean, target_mean)
 1278|                         if lpips_fn is not None:
 1279|                             lpips_val = lpips_fn(x1_pred, real_data[low_noise_mask], normalize=True).mean()
 1280|                             loss = loss + args.lpips_weight * lpips_val
 1281|                 else:
 1282|                     t = torch.randint(0, args.timesteps, (batch,), device=device, dtype=torch.long)
 1283|                     noisy_images, noise = diffusion.q_sample(real_data, t)
 1284|                     predicted = model(noisy_images, t, cond=cond, text_emb=text_emb_b)
 1285|                     if args.loss_type == "mse":
 1286|                         loss = F.mse_loss(predicted, noise)
 1287|                     else:
 1288|                         loss = F.l1_loss(predicted, noise)
 1289| 
 1290|                     # 颜色正则化 + LPIPS：只在低噪声 timestep 上约束