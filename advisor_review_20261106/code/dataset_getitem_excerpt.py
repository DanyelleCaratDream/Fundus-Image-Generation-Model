# train.py 第 186-325 行：FundusDataset（2026-11-06 同步增强修复后）
# 关键点：__getitem__ 顶部一次采样 do_flip/angle，同一组参数作用于 img/cond/weight

  186| # ============================================================
  187| class FundusDataset(Dataset):
  188|     """直接从文件夹读取眼底图，可选加载条件图（血管骨架）。"""
  189| 
  190|     def __init__(self, root, transform=None, cond_root=None, cond_transform=None, img_size=128,
  191|                  lesion_root=None, severe_names=None, feat_root=None, feat_k=2, augment=False):
  192|         self.transform = transform
  193|         self.cond_root = cond_root
  194|         self.lesion_root = lesion_root
  195|         self.severe_names = set(severe_names or [])
  196|         self.cond_transform = cond_transform or transform
  197|         self.img_size = img_size
  198|         # 同步空间增强（图像/条件图/病灶权重图共用一次 flip+rotate）。
  199|         # 2026-11-06 修复：此前随机翻转/旋转只作用于图像，cond 与权重图不跟随 → 系统性错位。
  200|         self.augment = augment
  201|         self.image_paths = []
  202|         if not os.path.isdir(root):
  203|             raise FileNotFoundError(f"数据集路径不存在: {root}")
  204| 
  205|         import glob
  206|         exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif",
  207|                 "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]
  208|         for ext in exts:
  209|             self.image_paths.extend(glob.glob(os.path.join(root, ext)))
  210|         self.image_paths = sorted(list(set(self.image_paths)))
  211| 
  212|         if len(self.image_paths) == 0:
  213|             raise ValueError(f"在 {root} 中没有找到任何图片！")
  214|         print(f"  找到 {len(self.image_paths)} 张图片")
  215| 
  216|         # 检索条件特征（C/A 路）：feats.npy + names.json，训练时做类内 Dirichlet 重心采样
  217|         self.feat_mat = None
  218|         self.feat_idx = None
  219|         self.feat_nbrs = None
  220|         self.feat_dim = 0
  221|         if feat_root:
  222|             fr = Path(feat_root)
  223|             X = np.load(fr / "feats.npy")
  224|             names = json.load(open(fr / "names.json", encoding="utf-8"))
  225|             name2i = {os.path.splitext(n)[0]: i for i, n in enumerate(names)}
  226|             Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
  227|             Xs = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-8)
  228|             self.feat_dim = int(Xs.shape[1])
  229|             self.feat_mat = torch.from_numpy(Xs.astype(np.float32))
  230|             labels = np.array([1 if os.path.splitext(n)[0] in self.severe_names else 0 for n in names])
  231|             sim = Xs @ Xs.T
  232|             self.feat_idx = [name2i.get(os.path.splitext(os.path.basename(p))[0])
  233|                              for p in self.image_paths]
  234|             self.feat_nbrs = []
  235|             for idx in self.feat_idx:
  236|                 if idx is None:
  237|                     self.feat_nbrs.append([]); continue
  238|                 cand = [j for j in range(len(names)) if labels[j] == labels[idx] and j != idx]
  239|                 order = sorted(cand, key=lambda j: -sim[idx, j])[:feat_k]
  240|                 self.feat_nbrs.append(order)
  241|             print(f"  检索特征: dim={self.feat_dim}, 覆盖 {sum(i is not None for i in self.feat_idx)}/{len(self.image_paths)}")
  242| 
  243|     def __len__(self):
  244|         return len(self.image_paths)
  245| 
  246|     def __getitem__(self, idx):
  247|         img = Image.open(self.image_paths[idx]).convert("RGB")
  248|         # 同步空间增强：一次采样 flip/angle，图像/条件图/权重图共用（2026-11-06 修复错位）
  249|         do_flip = bool(torch.rand(1).item() < 0.5)
  250|         angle = (torch.rand(1).item() * 2.0 - 1.0) * 15.0
  251|         if self.augment:
  252|             if do_flip:
  253|                 img = TF.hflip(img)
  254|             img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
  255|         if self.transform:
  256|             img = self.transform(img)
  257| 
  258|         # 条件图（血管骨架）
  259|         cond = None
  260|         if self.cond_root:
  261|             basename = os.path.splitext(os.path.basename(self.image_paths[idx]))[0]
  262|             cond_path = os.path.join(self.cond_root, basename + ".png")
  263|             if os.path.isfile(cond_path):
  264|                 if _HAS_CV2:
  265|                     cond_cv = cv2.imread(cond_path, cv2.IMREAD_GRAYSCALE)
  266|                     if cond_cv is not None:
  267|                         cond_cv = cv2.resize(cond_cv, (self.img_size, self.img_size))
  268|                         cond = torch.from_numpy(cond_cv).float() / 255.0 * 2.0 - 1.0
  269|                         cond = cond.unsqueeze(0)  # [1, H, W]
  270|                 else:
  271|                     cond_pil = Image.open(cond_path).convert("L").resize((self.img_size, self.img_size))
  272|                     cond = (torch.from_numpy(np.array(cond_pil)).float() / 255.0 * 2.0 - 1.0).unsqueeze(0)
  273|             if cond is None:
  274|                 cond = torch.zeros(1, self.img_size, self.img_size)
  275|             if self.augment:
  276|                 if do_flip:
  277|                     cond = torch.flip(cond, dims=[-1])
  278|                 cond = TF.rotate(cond, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=-1.0)
  279| 
  280|         basename = os.path.splitext(os.path.basename(self.image_paths[idx]))[0]
  281|         # 文本标签：severe 名单内=1，其余=0（仅 --use_text_cond 时使用）
  282|         tid = torch.tensor(1 if basename in self.severe_names else 0, dtype=torch.long)
  283| 
  284|         # 检索特征：anchor 偏置 + Dirichlet 类内重心采样
  285|         feat = None
  286|         if self.feat_mat is not None:
  287|             fi = self.feat_idx[idx]
  288|             if fi is None:
  289|                 feat = torch.zeros(self.feat_dim)
  290|             else:
  291|                 nbrs = self.feat_nbrs[idx]
  292|                 if nbrs:
  293|                     w_self = float(np.random.uniform(0.6, 0.9))
  294|                     w = np.random.dirichlet(np.ones(len(nbrs))) * (1.0 - w_self)
  295|                     v = self.feat_mat[fi] * w_self
  296|                     for j, wj in zip(nbrs, w):
  297|                         v = v + self.feat_mat[j] * float(wj)
  298|                     feat = v
  299|                 else:
  300|                     feat = self.feat_mat[fi].clone()
  301| 
  302|         # 病灶权重图（可选）：0=非病灶 → 权重 1，255 → 权重 1+λ
  303|         if self.lesion_root:
  304|             wp = os.path.join(self.lesion_root, basename + ".png")
  305|             w = torch.zeros(1, self.img_size, self.img_size)
  306|             if os.path.isfile(wp):
  307|                 if _HAS_CV2:
  308|                     wv = cv2.imread(wp, cv2.IMREAD_GRAYSCALE)
  309|                     if wv is not None:
  310|                         wv = cv2.resize(wv, (self.img_size, self.img_size))
  311|                         w = torch.from_numpy(wv.astype(np.float32) / 255.0).unsqueeze(0)
  312|                 else:
  313|                     wv = np.array(Image.open(wp).convert("L").resize((self.img_size, self.img_size)))
  314|                     w = torch.from_numpy(wv.astype(np.float32) / 255.0).unsqueeze(0)
  315|             if self.augment:
  316|                 if do_flip:
  317|                     w = torch.flip(w, dims=[-1])
  318|                 w = TF.rotate(w, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
  319|             if feat is not None:
  320|                 return img, cond, w, feat, idx
  321|             if self.severe_names:
  322|                 return img, cond, w, tid, idx
  323|             return img, cond, w, idx
  324|         if feat is not None:
  325|             return img, cond, feat, idx