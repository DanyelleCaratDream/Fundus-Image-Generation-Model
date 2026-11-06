# metrics_common.py：无偏 KID 实现（2026-11-06）

  122| def polynomial_kernel(X, Y, degree=3, coef0=1.0, gamma=None):
  123|     if gamma is None:
  124|         gamma = 1.0 / X.shape[1]
  125|     return (gamma * X @ Y.T + coef0) ** degree
  126| 
  127| 
  128| def _kid_mmd_unbiased(kk, ll, kl):
  129|     """无偏 MMD^2：核矩阵求和时排除对角线（样本与自身的核项）。"""
  130|     m, n = kk.shape[0], ll.shape[0]
  131|     kk_sum = (kk.sum() - np.trace(kk)) / (m * (m - 1))
  132|     ll_sum = (ll.sum() - np.trace(ll)) / (n * (n - 1))
  133|     kl_sum = kl.sum() / (m * n)
  134|     return kk_sum + ll_sum - 2 * kl_sum
  135| 
  136| 
  137| def compute_kid_stats(real_feats, fake_feats, degree=3, coef0=1.0,
  138|                       num_subsets=100, subset_size=100, seed=0):
  139|     """KID: 无偏核距离（Bińkowski et al. 2018）。
  140| 
  141|     无偏估计（排除核矩阵对角）；子集自助采样 num_subsets 次；
  142|     返回均值与标准差。子集大小自动收缩到不超过较小样本集的一半。
  143|     """
  144|     m = min(len(real_feats), len(fake_feats))
  145|     if m < 2:
  146|         raise ValueError("KID 需要至少 2 个样本")
  147|     subset_size = max(2, min(subset_size, m // 2))
  148|     rng = np.random.RandomState(seed)
  149|     vals = []
  150|     for _ in range(num_subsets):
  151|         ri = rng.choice(len(real_feats), subset_size, replace=False)
  152|         fi = rng.choice(len(fake_feats), subset_size, replace=False)
  153|         kk = polynomial_kernel(real_feats[ri], real_feats[ri], degree, coef0)
  154|         ll = polynomial_kernel(fake_feats[fi], fake_feats[fi], degree, coef0)
  155|         kl = polynomial_kernel(real_feats[ri], fake_feats[fi], degree, coef0)
  156|         vals.append(_kid_mmd_unbiased(kk, ll, kl))
  157|     return float(np.mean(vals)), float(np.std(vals))
  158| 
  159| 
  160| def compute_kid(real_feats, fake_feats, degree=3, coef0=1.0,

# evaluate() 中的写回
  360|     real_feats = inc(real)
  361|     fake_feats = inc(fake)
  362| 
  363|     results = {}
  364|     results["fid"] = compute_fid(real_feats, fake_feats)
  365|     results["kid"], results["kid_std"] = compute_kid_stats(real_feats, fake_feats)
  366|     results["mmd"] = compute_mmd(real_feats, fake_feats)
  367|     results["is"] = compute_is(fake, device)
  368|     results.update(compute_prdc(real_feats, fake_feats))  # precision/recall/density/coverage
  369|     results["one_nn"] = compute_1nn_test(real_feats, fake_feats)
  370|     results["ms_ssim"] = compute_msssim(fake, device)