# Anemoi-D2 复现可行性评估 + Phase B 重排（2026-06-18）

> Add-on to `sota_baseline_survey.md` (v1, 2026-06-16). User on 2026-06-18 asked to add **Anemoi-D2** (arXiv 2604.03303) to the SOTA-replication short-list and re-rank.

---

## 1. Anemoi-D2 真相 — 不能复现，但能 cite

### 论文 verify 结果

| 项 | 真实状态 |
|---|---|
| arxiv ID | **2604.03303** ✅ |
| 标题 | "Downscaling weather forecasts from Low- to High-Resolution with Diffusion Models" |
| 作者 | Dumont Le Brazidec, Lang, Leutbecher, Raoult, Mertes, Pinault, Tsiringakis, Maciel, Prieto Nemesio, Polster, O Brien, Chantry（ECMWF 团队） |
| 论文标题里**没有 "D2" 字样** | User 记忆的"D2"可能是 ECMWF 内部命名或别名 |
| 任务 | **global atmospheric downscaling**（不是 nowcasting） |
| 训练数据 | ECMWF IFS reforecast pairs（非公开数据集） |
| 公开代码 | ❌ 未在 anemoi-core 仓库 / 也未在 arxiv 给 GitHub URL（已 WebFetch verify） |
| 公开 ckpt | ❌ |
| 硬件资源 | 未声明，ECMWF 业务级（数百 GPU 起步） |

### 与我们任务的匹配度

- **任务不匹配**：downscaling 是空间超分（低分辨率 → 高分辨率），我们要 temporal forecasting（过去 → 未来）。两个任务的输入输出 schema 都不同
- **数据不匹配**：IFS reforecast 是全球网格 reanalysis，我们是华北 1 km 雷达
- **代码闭源**：没有 reference implementation
- **训练资源不可达**：业务级 ECMWF 集群我们没

### 复现可行性判断

**不可行**。跟之前调研中的 FusionCast / MAG-Net / VMU-Diff 同档——paper 真实存在，但**无公开代码 + 私有数据 + 算力远超我们**，公平 train-from-scratch 不可能。

### 论文里能怎么用

- **§4 Discussion** 里 cite 它作为"diffusion-based atmospheric downscaling 的代表性工作"
- 跟 PreDiff / CasCast / DiffCast 同框讨论"我们没走 diffusion 路径的原因"
- **不进 Phase B baseline 表**——只 cite，不重训

---

## 2. Phase B 重排（结合 mm/h narrative 反转）

### 2.1 新背景

mm/h 重训完成后，我们 5 个模型的 holdout 显示：
- ab1 → ab2 → ab3 **不再单调**（event_test 上 ab3 反而最差）
- ab3b 的 CSI@30 "collapse" **不再成立**（ab3b CSI@30 比 ab3 高）
- 整个 narrative 需要重写

→ **SOTA baseline 现在的首要价值不再是"vs published methods"**，而是**给我们 5 模型的绝对 skill 提供外部校准点**。

### 2.2 推荐执行顺序（按 ROI）

| 优先级 | Baseline | 真实成本 | 价值 | 推荐 |
|---|---|---|---|---|
| **🥇 P0** | **pysteps STEPS** | 0 GPU 训练 / 1 天 wrapper 工程 | 经典统计基线，给 DL 模型一个 "下界" 参考点。所有 nowcasting paper 标配 | ✅ 立即做 |
| **🥈 P1** | **Earthformer** | ~3-4 天工程 + ~36h GPU 训练 | 最干净的 deterministic transformer codebase，论文最有 anchor 价值 | ✅ 做 |
| **🥉 P2** | **exPreCast (Song et al. ICLR 2026)** | ~3-5 天工程 + ~24h GPU | 已 cite，跟我们 CDU 出自同源；repo 是个体复现需小心 | conditional：repo 健康度 verify 通过再做 |
| P3 | DGMR (openclimatefix port) | ~5-7 天工程 + ~48h GPU | Nature anchor，但 GAN 调参不稳，且 5min/256×256 vs 我们 6min/661×701 需大改 | 时间余裕再做 |
| P4 | CasCast deterministic stage | ~3 天工程 + ~24h GPU | 2024 SEVIR SOTA；可作 PreDiff 替代项 | 时间余裕再做 |
| **❌ NO** | **Anemoi-D2** | 不可行 | — | 仅 §4 cite，不重训 |
| ❌ NO | NowcastNet | 官方 thuml repo 404 | — | 仅 §4 cite，不重训 |
| ❌ NO | FusionCast / MAG-Net / VMU-Diff | 全闭源 | — | 仅 §4 cite，不重训 |

### 2.3 阶段拆解（约 1.5-2 周）

| 阶段 | 工作 | ETA |
|---|---|---|
| B0 | 建 `pluvian/baselines/` 框架 + pysteps wrapper + 在 our event_test/test_robust 上跑 pysteps | 1 天 |
| B1 | Earthformer dataloader 适配（661×701，12→18 frames，mm/h）+ 单卡 patch 训练 | 4 天 |
| B2 | Earthformer + ERA5 multimodal extension（手工加 input channel） | 2 天 |
| B3 | exPreCast sanity check (`tony890048/exPreCast` README quick-start 跑 5 min) → 决定要不要继续 | 0.5 天判断 + 3-5 天复现 |
| B4 | 整合表 + 论文 §3.5 baseline 比较节 + 第三轮 reviewer | 2 天 |

---

## 3. 立刻能反思的事 — 复现别人的过程顺便照镜子

User 提到"复现别人的过程顺便可以反思我们的问题"——已经从 Anemoi-D2 调研中得到几条 reflection：

| 我们的实践 | ECMWF Anemoi 项目 | 学到什么 |
|---|---|---|
| 单个仓库（pluvian/）含训练+评测+图全部脚本 | 拆成 anemoi-core / anemoi-graphs / anemoi-training / anemoi-datasets 四个独立 package | 我们 scripts/ 已经 30+ 文件，未来该考虑模块化 |
| 单一 entry-point train.py 写 ablation 开关 | 用 Hydra 配置 + plugin 机制 | 我们 configs/ 已经 10+ 个 ablation YAML 走类似路线，OK |
| 论文 ID 命名 ab1/ab2/ab3 | 用 dataset descriptor + model descriptor 命名 | 我们命名能追溯到 config，OK |
| 单位事故 dataset 输出 dBZ 但 paper 写 mm/h | ECMWF 框架强类型化 input/output（带 unit metadata） | 这是真痛点：我们 dataset 没有 unit annotation，论文 unit 跟代码 unit 脱节才导致 Phase 7e 事故 |

→ **可执行 takeaway**：未来在 dataset 输出加 unit attribute，eval 脚本里强制 sanity check 单位一致。但**不在论文 scope 内**，留作工程债。

---

## 4. 风险登记

1. **Anemoi-D2 在论文 §4 cite 时措辞要谨慎** — 不要写"vs Anemoi-D2 we improve X%"，写"Anemoi-D2 [Dumont Le Brazidec 2026] uses diffusion for downscaling, a different task from ours"
2. **如果 reviewer 坚持要看 Anemoi-style baseline** — 我们诚实说"closed source"。这是 review honest 题，跟 FusionCast 同档
3. **exPreCast tony890048 repo 风险**（之前调研里就标过）— 6 commits 个体仓库，要 sanity check
4. **pysteps wrapper 不能纯依赖** — 它是统计基线，performance ceiling 必然低于 DL，论文里要明确说"non-DL reference"，不能当 strong baseline 用来打 我们的脸

---

> **下一步**（等你拍板）：
> 1. 我立刻启动 pysteps B0（CPU，1 天工程，零 GPU 占用），不影响 bootstrap CI 后处理
> 2. 然后启动 Earthformer 调研 + dataloader 适配（最大工程量）
> 3. exPreCast 单独 sanity 之后决定
> 4. Phase 7e bootstrap CI 在 GPU0 后台跑（独立 track）
