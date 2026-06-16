# Pluvian SOTA Baseline Survey (2026-06-16)

> Reviewer round-1 response prep. Goal: 在 manifest split 上 train-from-scratch 跑得动、能与 ab1（radar-only）/ ab3（radar+PWV+ERA5）形成 fair compare 的 baseline 候选清单与执行计划。
> 验证策略：每个候选都做了 arxiv ID 和 GitHub URL 的 WebFetch 真实性核对；查不到的明确标注 `[VERIFY]`。

## 1. Background and constraints

| 维度 | 取值 |
| --- | --- |
| 雷达域 | 华北 661×701，0.01°≈1 km，36–42.6°N，113–120°E |
| 时间分辨率 | 6 min |
| 时序窗口 | 12 帧历史 (72 min) → 18 帧预报 (108 min) |
| 训练集 | 74 天 / 1 664 windows |
| 训练协议 | 60 epoch, batch=1+grad_accum=4 (effective 4), bf16, AdamW lr=3e-4 cosine |
| 硬件 | 单 GPU RTX 4090 24 GB（4 卡可并行多 baseline） |
| 公平性硬要求 | 必须在 manifest split 上 train-from-scratch；不接受公开 ckpt 跨域直接 inference |
| Multimodal 输入接口 | radar + GNSS-PWV (sparse tokens) + ERA5 (32 channels)；multimodal baseline 必须能接 PWV+ERA5 |
| Eval | CSI@{1,5,10,30} mm/h（full-set），FSS@{3,11} px，MAE，paired bootstrap CI |

这些约束等价于：

- 任何要求 ≥ 8 卡或 ≥ 72 h 训练的方法直接淘汰
- 任何 hard-coded SEVIR 384×384 / 256×256 网格、看上去无法换 661×701 输入的方法要标"重写 dataloader"成本
- horizon=18 frames 与多数论文（SEVIR 12 frames / DGMR 18 帧但 256×256 / NowcastNet 20 帧）需要做 sanity check
- 单 GPU 24 GB VRAM 上跑 661×701×30 帧的注意力类方法（Earthformer/CasCast/PreDiff）几乎一定要走 patch / latent 模式

## 2. Candidates surveyed

### A1. Radar-only nowcasting

#### DGMR (Skillful Nowcasting GAN, Ravuri 2021)
- 论文：Skillful Precipitation Nowcasting using Deep Generative Models of Radar，Ravuri et al., *Nature* 2021，arXiv:[2104.00954](https://arxiv.org/abs/2104.00954) — 已 WebFetch 核对，作者列表 "Suman Ravuri and 19 other authors" 一致。
- Codebase：DeepMind 官方只放 inference TF code + ckpt（英国 1 km 网格）；社区 PyTorch Lightning 重写 [openclimatefix/skillful_nowcasting](https://github.com/openclimatefix/skillful_nowcasting)（292 stars，2026-05 更新）已 WebFetch 验证存在。
- 输入/输出：4 帧 × 256×256 → 18 帧（90 min @ 5 min）。原始网格 1 km，receptive field 与我们项目网格匹配度好。
- 痛点：英国 1 km 单位 mm/h；切到中国数据要重写 dataloader、调输入帧数 12，horizon 18 ≈ 108 min（5→6 min 步长差异要核对）。GAN 训练对 single-GPU bs 要求高，社区版默认 effective bs=16。

#### NowcastNet (Zhang et al., *Nature* 2023)
- 论文：Skilful nowcasting of extreme precipitation with NowcastNet, Zhang et al., *Nature* 619, 526–532. 因 nature.com 跳认证，没法 WebFetch 全文，但 DOI [10.1038/s41586-023-06184-4](https://doi.org/10.1038/s41586-023-06184-4) 公认存在。
- Codebase：**官方 thuml/NowcastNet repo 我尝试访问返回 404**（GitHub API search 也未给出 thuml 命名空间下结果），目前可见的两个公开实现：
  - [VioletsOleander/nowcastnet-rewritten](https://github.com/VioletsOleander/nowcastnet-rewritten)（11 stars，inference-only，无训练脚本）
  - [Akashicf/ChaoXiNet](https://github.com/Akashicf/ChaoXiNet)（10 stars，README 写明是 "NowcastNet 复现"，含 train_*.py 脚本）
- 公平比较风险：官方训练代码缺失会拖慢复现，对 reviewer 而言 "复现版 NowcastNet" 的 fair compare 可信度<原版。`[VERIFY]` 是否还有 Code Ocean capsule（capsule/3935105）开放下载，我抓 codeocean 时返回 403。
- 痛点：NowcastNet 是 GAN+evolution，2 km USA-MRMS 网格，重训前要把 evolution module 改成 1 km 1 km / 6 min 步长。

#### MetNet-3 (Andrychowicz et al., 2023)
- 论文：Deep Learning for Day Forecasts from Sparse Observations, arXiv:[2306.06079](https://arxiv.org/abs/2306.06079) — 已核对。Google Search 在用的运营模型。
- 官方代码：**未释出**。社区 [lucidrains/metnet3-pytorch](https://github.com/lucidrains/metnet3-pytorch) 已核对存在（PyTorch 重写，MIT）。
- 公平比较风险：极高。MetNet-3 是 24 h 预报，6 min 步长 1 km 网格设定，但模型容量与训练算力（论文级 TPU 集群）远超我们 4×4090，社区版能跑但训练曲线/性能与原版不可比。
- 建议：**只作为论文里的 reference 引用**，不在 baseline column 比。

#### Earthformer (Gao et al., NeurIPS 2022)
- 论文：arXiv:[2207.05833](https://arxiv.org/abs/2207.05833)，作者 Gao, Shi, Wang, Zhu, Wang, Li, Yeung。已核对。
- Codebase：[amazon-science/earth-forecasting-transformer](https://github.com/amazon-science/earth-forecasting-transformer) — 已 WebFetch，含 SEVIR 训练脚本、ckpt、配置文件，结构清晰。
- 输入/输出：默认 SEVIR 13 → 12 frames @ 384×384（5 min），cuboid attention。
- 重训成本：dataloader 改 661×701×12→18 + 6 min 步长。Cuboid attention 对 661×701 全图直接跑会 OOM，需要分 tile / 改 cuboid 大小。**对 single-GPU 24 GB 可行**，是我们最确定能跑的强 baseline。

#### PreDiff (Gao et al., NeurIPS 2023)
- 论文：PreDiff: Precipitation Nowcasting with Latent Diffusion Models. arXiv 实际 ID `[VERIFY]`（[2307.10422](https://arxiv.org/abs/2307.10422) 大概率，没有 WebFetch 单独验证），repo 信息已 WebFetch。
- Codebase：[gaozhihan/PreDiff](https://github.com/gaozhihan/PreDiff) — 官方实现，PyTorch Lightning，SEVIR-LR 128×128 训练。已核对。
- 输入/输出：7 → 6 frames @ 128×128 latent。
- 痛点：分辨率太低（128×128）。要喂 661×701 必须先 train 一个新的 VAE。PreDiff 自身分三阶段（VAE → latent EarthFormer-UNet → knowledge alignment），三步串行，单卡时间预算可能爆。
- 价值：能做 ensemble + 概率预报，论文里讲 "我们提供 deterministic ab1 + 概率 baseline"。

#### exPreCast (Song, Chang, Hong, ICLR 2026)
- 论文：Extreme Weather Nowcasting via Local Precipitation Pattern Prediction, arXiv:[2602.05204](https://arxiv.org/abs/2602.05204)。**作者修正：Song / Chang / Hong**（已踩过坑，论文里别再写 Yang）。
- Codebase：[tony890048/exPreCast](https://github.com/tony890048/exPreCast) — 已 WebFetch，README 自称 "Official implementation of the ICLR'26 paper"，含 dataset.py / train.py / loss.py / metrics.py，引用 KMA radar / SEVIR / MeteoNet 数据集。仓库 commits 数较少（6 commits），代码新鲜度警告。
- 价值：deterministic、效率高、定位"极端事件"，刚好对齐我们 §4 论点。已经在论文中 cite。
- 重训：直接以 SEVIR loader 为模板改 661×701，预计成本中等。

#### pysteps STEPS (Pulkkinen 2019, ensembles)
- 论文：Pulkkinen et al. 2019, GMD, "Pysteps: an open-source Python library for probabilistic precipitation nowcasting (v1.0)"。
- Codebase：[pySTEPS/pysteps](https://github.com/pySTEPS/pysteps) — 已核对，活跃维护，2026-05-14 v1.21.1 release。
- 价值：经典 optical-flow + AR(p) 统计基线，跑得快，**所有 nowcasting paper 标配**。我们应该当作"非 DL baseline"列。
- 重训成本：无需训练（即时预报），只需要 wrap 我们 661×701 网格做 inference。

### A2. Multimodal nowcasting（关键查找）

#### LDCast (Leinonen et al., 2023)
- 论文：Latent diffusion models for generative precipitation nowcasting with accurate uncertainty quantification, arXiv:[2304.12891](https://arxiv.org/abs/2304.12891)。已核对，作者 Leinonen, Hamann, Nerini, Germann, Franch (MeteoSwiss)。
- Codebase：[MeteoSwiss/ldcast](https://github.com/MeteoSwiss/ldcast) — 已核对，"Latent diffusion for generative precipitation nowcasting"。
- Modality：核心 radar-only，但 latent diffusion 框架可外接条件，与 ERA5/PWV 集成路径相对干净。`[VERIFY]` 仓库默认 config 是否本身就有 NWP conditioning 入口。

#### CasCast (Gong et al., 2024)
- 论文：CasCast: Skillful High-resolution Precipitation Nowcasting via Cascaded Modelling, arXiv:[2402.04290](https://arxiv.org/abs/2402.04290)，已核对。
- Codebase：[OpenEarthLab/CasCast](https://github.com/OpenEarthLab/CasCast) — 已 WebFetch，含 deterministic（Earthformer 子模块）+ autoencoder + diffusion 三阶段训练 / 推理脚本，pre-trained weights 提供。
- Modality：**radar-only**（SEVIR-VIL），但级联设计与 PreDiff 类似，外挂 ERA5 conditioning 工程量可控。
- 价值：2024 年 SEVIR 上的强 SOTA，对我们 ab1 是有威胁的对比对象。

#### CasCast / FusionCast / MAG-Net / VMU-Diff（multimodal 真正贴近我们项目的）
- **FusionCast** (Wang et al., 2026), arXiv:[2603.13298](https://arxiv.org/abs/2603.13298)。已核对。**Modality = 历史 PWV (GNSS 反演) + 历史 radar QPE + 未来 radar prior**。**直接对应我们 ab3 的 PWV 通道**，narrative 价值极高。
  - Codebase：**未释出**（页面只列了 CatalyzeX / Papers with Code 链接占位，没有 GitHub URL）。`[VERIFY]` 是否在 ICLR/AAAI 接收后会公开。
- **MAG-Net** (Chen et al., 2026), arXiv:[2604.02818](https://arxiv.org/abs/2604.02818)。已核对。Modality = radar + 静止卫星 (IR 10.8 / WV 7.1 / BTD)。Codebase status `[VERIFY]`，未在 paper 页找到链接。
- **VMU-Diff** (Shi et al., 2026), arXiv:[2605.14597](https://arxiv.org/abs/2605.14597)。已核对，多源数据 (radar + multi-band satellite) + diffusion。Codebase 未给出。
- **WF-UNet** (arXiv:[2302.04102](https://arxiv.org/abs/2302.04102))：精度 vs ab3 大概率不强，但作为简单 multimodal baseline 可备选 `[VERIFY]` 代码。

### A3. 同领域负结果 / 物理损失诊断

我没有找到一篇主流论文专门论证 "PDE/water-budget loss 对极端事件有副作用"。能拿来做 §4.2 论点支撑的，最相关的是：

- **CasCast / PreDiff / DiffCast** 这一系：所有 diffusion 派系都吐槽 deterministic L2 loss 对 mean blurry，但他们的解法是 stochastic generation 而不是直接说 "physics loss bad"。
- **PostCast** (arXiv:[2410.05805](https://arxiv.org/abs/2410.05805))：unsupervised post-process diffusion 来 debblur deterministic 预测，间接证明 hard physics constraint 不一定救得了 extreme tails。
- **DiffCast** (Yu et al., CVPR 2024, arXiv:[2312.06734](https://arxiv.org/abs/2312.06734))：[DeminYu98/DiffCast](https://github.com/DeminYu98/DiffCast) — 已核对。残差 diffusion 处理 deterministic blurry。

> **诚实结论**：缺乏一篇 "physics loss hurt extremes" 的负结果论文。论文 §4.2 论点要靠我们自己的 ablation 数据撑，不能宣称有外部 reference 直接背书。

## 3. Feasibility × value scoring table

打分维度（0–3）：1) 公开度 2) 接口适配 3) 公平比较可信度 4) narrative 相关度 5) GPU 成本 (≤24 h=3, 24–72 h=2, >72 h=1, 不实际=0)

| 候选 | 公开度 | 接口 | 公平 | narrative | GPU | 合计 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pysteps STEPS | 3 | 3 | 3 | 2 | 3 | **14** | 即时预报，必跑 |
| Earthformer | 3 | 2 | 3 | 2 | 2 | **12** | 强 deterministic baseline |
| exPreCast | 2 | 2 | 2 | 3 | 3 | **12** | 已 cite，narrative 直接对齐 §4 |
| CasCast | 3 | 1 | 3 | 2 | 1 | **10** | 三阶段训练，单卡时间偏紧 |
| DGMR (社区版) | 3 | 1 | 2 | 2 | 1 | **9** | GAN bs / 时间步长不一致风险 |
| LDCast | 3 | 1 | 2 | 2 | 1 | **9** | latent diffusion，需训 VAE |
| PreDiff | 3 | 1 | 2 | 2 | 1 | **9** | 128×128 太低，要重训 VAE |
| NowcastNet (复现) | 1 | 1 | 1 | 2 | 1 | **6** | 官方 repo 缺失，社区版可信度低 |
| MetNet-3 | 2 | 0 | 0 | 2 | 0 | **4** | 训练算力不可达 |
| FusionCast | 0 | — | — | 3 | — | **0+** | 代码未公开，只能引用 |
| MAG-Net / VMU-Diff | 0 | — | — | 2 | — | **0+** | 同上 |

> 注：合计只是粗排序工具；最终 short-list 还结合工程难度和并行能力。

## 4. Short-list (top 5) detailed cards

### 1. pysteps STEPS（必跑，先跑）
- 论文：Pulkkinen et al. 2019, GMD（pysteps v1.0）。
- Code: https://github.com/pySTEPS/pysteps （已核对，2026-05-14 v1.21.1）。
- 重训路径：**无需训练**。`pysteps.nowcasts.steps.forecast` 直接喂 661×701 dBZ 序列，输出 18 帧。注意 dBZ → mm/h Z–R 换算要和我们 manifest 对齐。
- Multimodal：N/A（纯 radar 统计基线）。
- GPU 成本：CPU only，每 window 秒级。
- 为什么跑：所有 nowcasting paper 的标配统计基线；reviewer 一定问；能直接验证我们 ab1 的"DL > 统计"的下界。

### 2. Earthformer（强 deterministic 比对）
- 论文：arXiv:2207.05833，NeurIPS 2022。
- Code: https://github.com/amazon-science/earth-forecasting-transformer
- 重训路径：
  1. 写 `pluvian_dataset.py`，输出 (B, 12, H, W, 1) → (B, 18, H, W, 1)，对齐我们 manifest split
  2. config 改 `cuboid_size`、`input_shape=(12,661,701,1)`、`target_shape=(18,661,701,1)`
  3. 661×701 全图直接喂会爆 24 GB；改成 256×256 patch + overlap，或在 cuboid 第一层做 2× downsample
  4. loss 替换为我们的 intensity-stratified MAE + CSI 监控
- Multimodal 可能性：模型本身只接一个 channel；ERA5 32 通道可在 input embedding 维 concat（与 ab3 设计风格一致），PWV sparse token 难塞，需要单独 cross-attention head（工程量中等）。
- GPU 成本：单卡 24 GB 估 24–48 h（patch 模式）。
- 为什么跑：deterministic transformer SOTA 之一，与我们 ab1 对比公平、可信、reviewer 接受度高。

### 3. exPreCast（narrative 对齐"极端事件"）
- 论文：arXiv:2602.05204，ICLR 2026。
- Code: https://github.com/tony890048/exPreCast（仓库较新，6 commits，需谨慎）
- 重训路径：模仿 SEVIR loader 改 661×701；`config.py` 改输入帧数 12、horizon 18、6 min 步长；保留作者的 local pattern prediction loss。
- Multimodal：deterministic、轻量，加 ERA5 通道直接 concat 难度小；PWV sparse token 同样需要新增 head。
- GPU 成本：作者强调 "efficient"，单卡估 ≤24 h。
- 为什么跑：(a) 论文已经 cite 了它；(b) 它的"local precipitation pattern"动机和我们 §4 在极端事件上的论点强一致；(c) 是 2026 ICLR 最新，回应 reviewer "你们和最新 SOTA 比了吗"。

### 4. DGMR (openclimatefix 社区版)（GAN 概率比对）
- 论文：arXiv:2104.00954, *Nature* 2021。
- Code: https://github.com/openclimatefix/skillful_nowcasting（PyTorch Lightning，2026-05 更新）
- 重训路径：
  1. dataloader 改 661×701，input frames 4 → 12（DGMR 默认 4 帧 conditional context，能直接拓 12，注意改 G/D 通道维数）
  2. 6 min 步长（默认 5 min，scheduling 改即可）
  3. effective bs 必须维持 ≥8（GAN 稳定性），单卡 24 GB 上 grad accumulation 16
  4. **horizon 18 帧**：原代码 18 帧 256×256，661×701 必须切 patch
- Multimodal：原模型纯 radar；ERA5 可作为 conditioning vector 进 latent stack（与 LDM-cond 风格类似），但属于"小改"以上。
- GPU 成本：48–72 h，时间预算紧。
- 为什么跑：Nature 论文级权威；做 fair compare 时 reviewer 心里"DL nowcasting 鼻祖"那一栏才不空白。

### 5. CasCast（最新 SEVIR-class 强 baseline）
- 论文：arXiv:2402.04290。
- Code: https://github.com/OpenEarthLab/CasCast（含 train/eval scripts，pretrained 提供）
- 重训路径：
  1. 三阶段：deterministic Earthformer → autoencoder → CasFormer diffusion。可以先只跑 deterministic stage 做 ab1 对比，省一半时间
  2. 661×701 输入要重训 autoencoder（latent 48×48 → 改成 ~83×88）
  3. SEVIR list 替换为 manifest split list
- Multimodal：deterministic stage 可在 input concat ERA5 channels；diffusion stage 接 PWV 难度高
- GPU 成本：完整三阶段 ≥72 h；只跑 deterministic 段约 24–36 h
- 为什么跑：(a) 2024 上半年 SEVIR 上的 SOTA；(b) reviewer 看到 deterministic 阶段也能讲故事；(c) 后续如果做扩展到 v2，CasFormer 可作概率版升级。

### 备选：LDCast / PreDiff
- LDCast 适合做 "概率 baseline"，但代码工程化程度低于 CasCast；PreDiff 128×128 → 661×701 必须重训 VAE，时间预算危险。**只在 short-list 前 5 个有人挂掉时再上**。

## 5. Execution plan proposal (Phase B)

### 推荐 commit 规模：3 个 baseline + 1 统计 baseline = 4 条对比线
- **必跑**：pysteps STEPS、Earthformer、exPreCast
- **强烈建议**：DGMR-社区版（reviewer 对 Nature 的 anchor 反应大）
- **机动**：CasCast deterministic 段（看时间预算）

### 阶段拆解（外科手术式接入项目 `scripts/train.py`）

| 步骤 | 文件/动作 | 验证 | 完成标准 |
| --- | --- | --- | --- |
| B1 | 在 `src/baselines/` 下建 `pysteps_runner.py`，复用项目 `eval_pipeline` | 跑 1 个 window，看 18 帧 forecast 形状对 | `python -m baselines.pysteps_runner --window <id>` 输出 18×661×701 |
| B2 | clone earth-forecasting-transformer，加 `src/baselines/earthformer/dataset.py`（继承我们 dataset）、`config_pluvian.yaml` | 单 window 前向 + loss 反传不爆显存 | 1 epoch 训练能跑通 ≥10 step |
| B3 | clone exPreCast，写 `train_pluvian.py` 类比作者 `train.py`，复用 manifest split | 同 B2 | 同 B2 |
| B4 | clone openclimatefix/skillful_nowcasting，调 dataloader / G & D 输入维度 | 同 B2，重点看 GAN bs+grad accum | 1 epoch GAN 稳定不发散 |
| B5 | （机动）clone OpenEarthLab/CasCast deterministic 段 | 同 B2 | 同 B2 |
| B6 | 复用项目 `eval/csi_fss.py` & `paired_bootstrap.py` | 在每个 baseline 输出上跑评测 | 输出与现有 ab1/ab3 表格 schema 一致 |
| B7 | 在 `docs/experiments/sota_baseline_results.md` 落地表格 | 与 paper §3.4 表格对齐 | reviewer 可直接看 |

### 4×4090 上的并行编排

> 4 卡，每个 baseline 单卡训练为主，所以串行不如分卡并行。

| GPU | Day 0–2 | Day 2–4 | Day 4–6 |
| --- | --- | --- | --- |
| 0 | Earthformer | Earthformer 评测 + DGMR | DGMR 评测 |
| 1 | exPreCast | exPreCast 评测 | CasCast deterministic |
| 2 | pysteps（CPU 富裕，可放 GPU 0 idle 时段） | DGMR | CasCast 评测 |
| 3 | 多模态扩展实验 / ablation 备份 | 复跑 / 调参 | 复跑 / 调参 |

> 总时间预算 ≤ 7 天可拿到 3+1 baseline。如果加 CasCast 完整 diffusion 段，预算 ≤ 10 天。

### 项目内接入点（外科手术式）
- 数据：`src/data/manifest.py` 已经能输出 (12, 661, 701) → (18, 661, 701) 张量，所有 baseline 共用同一个 dataloader（保证 fair）
- Loss：项目已有 intensity-stratified MAE / CSI proxy，每个 baseline 替换它的官方 loss 文件即可
- Eval：`src/eval/{csi_fss, paired_bootstrap}.py` 已是项目级，所有 baseline 输出走同一 pipeline
- 训练入口：`scripts/train.py` 加 `--model {earthformer,exprecast,dgmr,cascast}` 分支，每个分支 import baseline 包装类

## 6. Honest risks and what we cannot do

1. **dBZ vs 项目雷达单位的 systematic error**：DGMR/NowcastNet/Earthformer 训练时输入 mm/h 或 VIL；我们的 dataloader 必须确保单位一致，否则 numerical scale 全错。这是 **B1 之前必须解决的 prerequisite**。
2. **MetNet-3 不可比**：单 GPU 训练规模与 Google TPU 集群相比差 ≥100×。社区版只能训出 toy 性能，不能拿来当 fair baseline，论文里只能作 reference。
3. **Multimodal baseline 几乎无 train-from-scratch 选项**：FusionCast、MAG-Net、VMU-Diff 全部 **未公开代码**。如果 reviewer 要求 "和 multimodal SOTA 比"，我们只能：(a) 在 deterministic baseline 上手工 concat ERA5 通道做"pseudo-multimodal"对比；(b) 在论文里诚实写 "公开代码缺失，只能 cite"。这一点必须提前向 reviewer 摊牌。
4. **NowcastNet 官方 repo 失踪**：访问 thuml/NowcastNet 返回 404，唯一可用版本是社区复现（10–11 stars，不完整 train pipeline）。如果要 cite，需写 "复现版本" 并降低权重；不能宣称比的是原版。
5. **DGMR 4 km vs 1 km mismatch**：DGMR 原始网格 1 km / 5 min（英国），我们 1 km / 6 min（华北）。空间分辨率匹配，但时间步长差 1 min，receptive field 推断会有 6/5≈1.2× 偏差。**必须在 paper 中明写：DGMR 我们重训而非直接用 ckpt**。
6. **没有 PDE-loss 负结果论文做 §4.2 背书**：物理损失对极端事件的副作用，目前看主要靠 diffusion-debias 间接论证，没有正面 ablation。论文里要靠我们的实验数据自己撑，引用范围是 PostCast / CasCast / DiffCast 这一系（讲 deterministic blurry → diffusion 救场）。
7. **`[VERIFY]` 项**：
   - PreDiff arxiv ID 我没单独 WebFetch 验证（取自 PreDiff GitHub README 的描述）
   - LDCast 的 NWP conditioning 是否原生支持，需要 clone 后看 config
   - exPreCast 仓库非常新（6 commits），训练曲线复现稳定性不确定
   - 几个 multimodal arxiv ID（2603.x / 2604.x / 2605.x）都是 2026 年 arxiv 编号，已 WebFetch 验证标题 + 摘要存在；但代码未公开
8. **算力余量**：4×4090 只有 24 GB/卡，CasCast diffusion / PreDiff 长期训练可能 OOM 或 epoch 时间过长；如发现 >72 h 立即撤回到 deterministic 段。

---

> 报告版本：2026-06-16 v1。下一步：得到用户确认 commit 哪几个 baseline 后，开 Phase B 任务卡。
