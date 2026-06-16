# Pluvian — 实验过程与决策记录（Process）

> 项目过程子文档 | 主文档：*Pluvian — 多模态深度学习短临降水预报*
>
> 这里记录所有踩过的坑、决策的推理、子实验的具体数据、reviewer concerns 的完整对策。**主文档读高层结论，本文档读"为什么 / 怎么发现的 / 数字到底是多少"。**

---

## §A. Phase 7e 单位事故（CRITICAL，2026-06-16）

### A.1 事故描述

在为 reviewer round-1 C3 (外部 baseline) 准备 SOTA 调研时，我们 verify 了几个公开 nowcasting repo 的输入单位（DGMR mm/h、Earthformer dBZ、exPreCast dBZ），过程中突然意识到要验证**我们自己的 dataset 单位**。

直接读 `pipeline/data_loader.py::NPJDataset.__getitem__`，发现：
- dataset 输出的 radar tensor 来自 `radar_io.load_radar_sequence`，是 xarray-decoded **raw dBZ**（典型范围 0–70 dBZ）
- **没有** `dbz_to_rainrate` 转换
- 但论文 §2.1 写 "all losses and metrics are in mm h⁻¹"
- 损失 `band_edges = [0.1, 1.0, 8.0, 30.0]` 标作 mm/h，被 `torch.bucketize` 直接 vs raw dBZ
- CSI 阈值 `(1.0, 5.0, 10.0, 30.0)` 同样直接 vs raw dBZ
- `band_centers = (0.0, 0.5, 4.5, 19.0, 50.0)` 在 model decoder 里注释 "mm/h"

**结论**：rain head 在 dBZ 上训了 60 epoch，但每个标签都说是 mm/h。

### A.2 数字层面的影响

CSI 是个对任意阈值都 well-defined 的指标，**数字本身没崩**。但论文叙事的 "extreme rain" 框定**完全错位**：

| 论文叙事 | 实际是 |
|---|---|
| CSI@30 = "extreme tail" | 30 dBZ ≈ 1.5 mm/h，按气象标准是 light-to-moderate rain |
| ab3b 抹平的"强对流核" | 实际只是 30 dBZ 的 cells，不是真极端核（>50 dBZ） |
| event_test ">30 mm/h" | 实际是 >30 dBZ（占 6.4% 像素，正常 rain coverage） |

mm/h 下真实分布（2023-07-29 全 30 帧）：max 103.8 mm/h，mean 0.54 mm/h，>30 mm/h 仅占 **0.04% 像素**。

### A.3 Reviewer 早就提示但我们没读懂

| Reviewer | Round | 标记 | 内容 |
|---|---|---|---|
| #1 | 1 | M3 | "Table 1 caption mixes mm h⁻¹ thresholds with a 30 dBZ contour reference in Fig 4" |
| #2 | 2 | m1 | "§3.3 mixes '30 dBZ contour' with CSI thresholds in mm h⁻¹" |

我们把它当 Fig 4 caption 一处不一致来修，**没意识到是整篇 unit 系统性错误**。这次发现是因为为 SOTA baseline 准备 input 单位对齐时主动审 dataset。

### A.4 修复

commit `300b9cb`：dataset `__getitem__` 加 `utils.dbz_to_rainrate`（China operational Z = 300 R^1.4）。

```diff
+ from . import utils
  ...
  radar, radar_mask, ... = radar_io.load_radar_sequence(start, self.n_frames)
  if meta.get("zero_day"):
      radar[:] = 0.0
+ # Phase 7e: convert raw dBZ -> mm/h to align with paper text
+ radar = utils.dbz_to_rainrate(radar)
```

### A.5 蕴含

- 5 个旧 ckpt（ab1, ab2, ab3, ab3b, ab3-cdu）**全部作废** — 它们是在 dBZ targets 上训的，喂 mm/h 会全乱
- 论文所有 metric 数字、figures、tables、paired CIs 都**变 stale**
- **必须 from-scratch 重训** 5 个模型
- Phase B (SOTA baseline) 阻塞 — baseline 必须在 mm/h 上训

### A.6 Phase 7e 重训进程（2026-06-16 22:09 UTC 启动）

GPU 资源问题：当时另一用户 ZhaoJiaming 占 8 张 GPU（streaming-VLM 训练），smoke test OOM。等 ~3 小时他训完，6 张 GPU 释放后启动。

5 路并行：

| GPU | Config | PID | 备注 |
|---|---|---:|---|
| 0 | `configs/ablation_1_p7d.yaml` | 2020763 | radar baseline |
| 3 | `configs/ablation_2_p7d_xcoreA.yaml` | 2020764 | +PWV |
| 4 | `configs/ablation_3_era5_p7c.yaml` | 2020765 | +ERA5 |
| 5 | `configs/ablation_3b_era5_budget_p7c.yaml` | 2020766 | +budget loss, weight=0.1 |
| 6 | `configs/ablation_3_cdu_p7d.yaml` | 2020767 | **from-scratch** (no warm-start) |

**CDU from-scratch 决策**：之前 CDU 是 warm-start from ab3 best.pt，但现在新 ab3 还在训，不能 warm-start。**意外好处**：自动解决 reviewer C6 / N1 的 warm-start vs from-scratch confound（reviewer 本来就要求做这个 control 实验）。

**初始 loss 健康性**：ab3b 的 budget loss 在 mm/h 单位下 ≈ 5×10⁻⁴（dBZ 时是 1×10⁻⁴），**5 倍增大**。这意味着 PDE 项在 mm/h 单位下提供更有意义的 gradient signal — 可能改变 §3.2 narrative。

ETA：~18-20 h wall-clock，预期 **2026-06-17 16-18 UTC**（北京 00:00-02:00）完成。

---

## §B. 五个模型的实验链路（按时序）

### B.1 ab1 / ab2 — Phase 7c（早期，已完成）

| 模型 | 时间 | 关键事实 |
|---|---|---|
| ab1 (radar) | 2026 早期 | radar-only baseline，60 epoch |
| ab2 (xcoreA) | 2026 早期 | 加 GNSS-PWV cross-attention，60 epoch |

ab2 的 PWV cross-attention design 经过 xcoreA / xcoreB / 默认 ab2 三个变体。论文用 **ab2_p7d (from-scratch)** 作为主线（CSI@30 = 0.283）。

### B.2 ab3 — Phase 7c（已完成）

加 ERA5 reanalysis stack（u, v, q, T × 8 hPa levels = 32 channels），通过 GatedAsymmFusion 融合。CSI@10 0.632 vs ab2 0.591 (+7%)。

### B.3 ab3b — Phase 7d（关键负结果）

ab3 + 物理 water-budget PDE 损失（continuity equation residual）。Loss form：
```
ε = ∂PWV/∂t + (Δp/g) · ∇·(qV) + P_predicted
loss_budget = HuberLoss(ε)
```
weight = 0.1，δ = 1e-4。

**关键观察**：bulk metric 比 ab3 全面更高（CSI@1 0.492 vs 0.470, MAE 4.10 vs 4.27），**但 CSI@30 暴跌 -28%**（0.291 → 0.210）。

### B.4 ab3-cdu — Phase 7d（架构 ablation）

借鉴 exPreCast (ICLR 2026)。**仅替换** decoder 的 `_UpBlock` 为 `_CDUUpBlock`，其他 (data / loss / optim) 完全一致。

之前 warm-started from ab3 best.pt（旧版本），新版 from-scratch。

### B.5 Budget weight sweep — Phase 7d（response to reviewer C5）

跑 weight ∈ {0.001, 0.01, 0.1} 三个变体（其他完全一致）。

**结果**（dBZ-era data）：

| weight | event_test CSI@30 | paired Δ vs ab3 |
|---|---:|---|
| 0 (ab3) | 0.291 | — |
| 0.001 | 0.276 | −0.015 [−0.023, +0.001] (n.s.) |
| 0.01 | 0.212 | −0.078 [−0.088, −0.055] |
| 0.1 (主) | 0.210 | −0.081 [−0.091, −0.057] |

**关键洞察**：
- weight ≥ 0.01 时 CSI@30 cost **饱和**（0.01 vs 0.1 paired CI 重叠）
- 只有 weight = 0.001 时 budget loss value ≈ 1×10⁻⁵（vs data loss ~10），物理项**已停止约束模型**

→ Trade-off 是物理约束实际起作用的权重区间内的 **structural property**，不是 hyperparameter mistuning。

但 reviewer N1 注意到：weight=0.001 在 bulk 指标上**反而比** weight=0.1 主 ab3b 更好（CSI@1 0.507 vs 0.492）。这是 cherry-picking 风险（详见 §C.N1）。

---

## §C. Reviewer Concerns 完整对策表

### C.1 Round-1 reviewer (8 concerns)

| ID | Concern | Status | 处理 |
|---|---|---|---|
| C1 | "Headline collapse 不被 reported uncertainty 支撑（marginal CIs overlap）" | ✅ Closed | 跑 paired bootstrap delta CI (n=2000)，event_test [−0.091, −0.057] / robust [−0.078, −0.019]，严格 < 0 |
| C2 | "§3.4 (CDU) is empty but Introduction sells it as a contribution" | ✅ Closed | CDU 训完后 §3.4 用真实数字写完，contribution 重定位为 "avoids trade-off, doesn't recover" |
| C3 | "No comparison against published nowcasters (DGMR/NowcastNet/Earthformer/MetNet-3)" | 🔄 In progress | Phase B SOTA baseline 计划，详见 §D |
| C4 | "Pooled-CSI novelty overstated" | ✅ Closed | §3.3 / §4.3 重写为 "diagnostic use of established neighbourhood verification (Ebert 2008; Mittermaier 2014)" |
| C5 | "ab3b failure attribution not unique — could be mis-tuned hyperparameter" | ✅ Closed | Budget weight sweep（详见 §B.5），证明 trade-off 在 weight ≥ 0.01 饱和 |
| C6 | "ab3-cdu warm-start 不公平 vs from-scratch ab1/ab2/ab3/ab3b" | ✅ Auto-closed | Phase 7e 重训自动 from-scratch |
| C7 | "Generalisability claim outruns the data" | ✅ Closed | Abstract / §1 (iii) softened to "hypothesise"; §4.5 显式 scope 4+8 天 |
| C8 | "CRPS == MAE in metric files (degenerate single-member)" | ✅ Closed | §2.4 删除 CRPS 解释；论文只报 MAE |
| M1 | Editorial HTML comments leak | ✅ Closed | 全部删除 |
| M2 | "monotonic" vs csi30 实际非单调 | ✅ Closed | 改为 "monotonic for CSI@1/5/10, FSS, MAE; non-monotone for CSI@30" |
| M3 | "30 dBZ contour vs mm/h thresholds 不一致" | 🔄 Phase 7e | **整篇 unit 系统性事故，正在重训**（详见 §A） |
| M4 | PWV 7 个 dropped stations 没列 ID | ⏳ Pending | 待补 supplementary |
| M5 | §1 cite 但 bibliography 没有的项 | ⏳ Pending | Bibliography pass 时统一处理 |

### C.2 Round-2 reviewer (3 new concerns)

| ID | Concern | Status |
|---|---|---|
| **N1** | weight=0.001 ab3b 在 bulk 指标 Pareto-dominate 主 weight=0.1 ab3b（CSI@1 0.507 vs 0.492，MAE 3.97 vs 4.10），暗示 warm-start confound — 部分 bulk gain 不是 budget loss 的功劳 | 🔄 Phase 7e 自动消除（5 模型都 from-scratch） |
| **N2** | §3.4 文字说 CDU "essentially neutral at extreme tail"，但 Fig 4 case study 显示 CDU 在该窗口实际把 46 dBZ 真实核压到 26 dBZ。无 per-window 方差报告 | ⏳ Pending — 需 per-window CSI@30 分布图 |
| **N3** | CDU 既然不"recover extreme"，contribution (iii) 实质是"换 decoder block 让 CSI@1 涨 0.018"，对 npj 这种结果导向期刊偏弱。建议要么 reframe 为 methods paper，要么 CDU 拆独立投稿 | ⏳ Pending — 等 mm/h 重训后 CDU 极端表现可能改变，再决定 |

---

## §D. SOTA Baseline 调研与 Phase B 计划（reviewer C3）

### D.1 调研方法

派独立 agent 做 nowcasting SOTA 全景调研（2026-06-16，commit `a11dfbe`，文件 `docs/experiments/sota_baseline_survey.md`）。每个候选都 verify arxiv ID + GitHub URL（吸取上次 exPreCast 作者写错为 Yang 的教训）。

### D.2 候选清单与评分

| 候选 | 公开度 | 接口适配 | 公平性 | narrative | GPU 成本 | 总分 |
|---|---|---|---|---|---|---|
| pysteps STEPS | 3 | 3 | 3 | 2 | 3 | **14** |
| Earthformer | 3 | 2 | 3 | 2 | 2 | **12** |
| exPreCast | 2 | 2 | 2 | 3 | 3 | **12** |
| CasCast | 3 | 1 | 3 | 2 | 1 | **10** |
| DGMR (社区版) | 3 | 1 | 2 | 2 | 1 | **9** |
| NowcastNet (复现) | 1 | 1 | 1 | 2 | 1 | **6** |
| MetNet-3 | 2 | 0 | 0 | 2 | 0 | **4** |

### D.3 Short-list 与执行计划

User 已确认 Tier 1+2：**pysteps + Earthformer + exPreCast + DGMR**。给每个 baseline 手工加 ERA5 channel 作为 multimodal extension（vs ab3 比较）。

执行阶段（约 2-3 周）：

| 阶段 | 工作 | 估时 |
|---|---|---|
| B0 | 建 `pluvian/baselines/` 框架 + pysteps wrapper | 1 天 |
| B1 | Earthformer radar-only on our data | 3-4 天 |
| B2 | Earthformer + ERA5（手工加 channel） | 2-3 天 |
| B3 | exPreCast radar-only + multimodal | 3-5 天 |
| B4 | DGMR radar-only + multimodal（最难） | 5-7 天 |
| B5 | 论文 §3.5 baseline 对比章节 + 第三轮 reviewer | 1-2 天 |

最终对比表预期 **11 行**（5 ours + 6 baselines）× 7 metrics × 2 splits + paired CI。

### D.4 诚实风险

- **Multimodal SOTA 几乎无可重训实现**：FusionCast / MAG-Net / VMU-Diff 全闭源；公平 multimodal 比对**做不到**，论文里只能 limitation 显式认领
- **NowcastNet 官方 thuml repo 返回 404**，社区复现 10-11 stars，可信度低
- **DGMR 时空步长不同**（5 min / 256×256 vs 我们 6 min / 661×701），重训前必须改 dataloader + receptive field
- **没有"PDE loss hurt extremes"的负结果论文做 §4.2 背书**，论点要靠我们自己的 ablation 撑

---

## §E. 决策时间线（按月-日）

| 日期 | 事件 | Commit |
|---|---|---|
| 2026 早期 | ab1 / ab2 初版训练 | (phase7c) |
| 2026-06-02~04 | ab3 / ab3b 训完 | (phase7c) |
| 2026-06-13 晚 | 接手项目，发现 ab3b 没评测，跑完发现 CSI@30 暴跌 -28% | — |
| 2026-06-14 | 写 pooled-CSI 诊断脚本，证明 ab3b 是 smoothing 不是 displacement。决策 ab3 = 主线，ab3b = negative result | — |
| 2026-06-14 晚 | CDU 实现 + 启动训练 | — |
| 2026-06-15 | 第一轮独立同行评审（8 concerns），跑 paired bootstrap CI 反驳 C1 | `415357e` 等 |
| 2026-06-15 晚 | CDU 训完，holdout eval 显示 CSI@30 持平 ab3。论文叙事重定位 | `750103f` |
| 2026-06-16 凌晨 | Budget weight sweep 训完，paired delta CI 完成 | `187663e` |
| 2026-06-16 中午 | 第二轮独立同行评审（3 new concerns），写 round-2 review | `3046e62` |
| 2026-06-16 中午 | SOTA baseline 调研 → Phase B 计划 | `a11dfbe` |
| 2026-06-16 14:00 | **发现单位事故**（dBZ vs mm/h），dataset 改动 commit | `300b9cb` |
| 2026-06-16 22:09 | **5 路并行重训启动**（mm/h 单位） | `316f1b9` |
| 2026-06-17 上午 | 重训预计完成 | TBD |

---

## §F. 已知 todo 详表（投稿前）

| 项 | 优先级 | 估时 | 等待条件 |
|---|---|---:|---|
| Phase 7e 重训完成 + 全套数字重出 | P0 | ~24 h wall-clock | 自动 |
| 第三轮独立同行评审（mm/h 数据） | P0 | 0.5 天 | 等 Phase 7e |
| 论文 §1–§4 数字全替换 + 单位 audit | P0 | 1 天 | 等 Phase 7e |
| Phase B 启动（pysteps wrapper） | P1 | 2-3 周 | 等 Phase 7e |
| Fig 1 / Fig S1 | P1 | 1 天 | 导师拍板风格 |
| Bibliography → BibTeX | P1 | 0.5 天 | 论文定稿前 |
| Reviewer N2 per-window CSI@30 分布 | P2 | 0.5 天 | Phase 7e 后 |
| Reviewer N3 framing 决策 | P2 | 协商 | Phase 7e 后看新数据 |
| 论文最终 polish | P2 | 1 天 | 投稿前 |

---

## §G. 资源与文件位置

| 资源 | 路径 |
|---|---|
| 仓库根 | `/data4/WuMingrui/TianJinyu/npj/pluvian` |
| 论文章节 | `docs/paper/{MAIN.md, section_1_introduction.md, ..., tables_1_2.md}` |
| Reviewer reports | `docs/paper/REVIEW_round{1,2}.md` |
| 完整决策 ledger | `docs/experiments/pluvian_phase7d_ledger.md` |
| SOTA 调研 | `docs/experiments/sota_baseline_survey.md` |
| 实验配置 | `configs/ablation_*.yaml` |
| 训练入口 | `scripts/train.py` |
| 评测入口 | `scripts/eval.py` + `scripts/eval_pooled_csi.py` + `scripts/eval_per_lead.py` + `scripts/bootstrap_delta_ci.py` |
| Bootstrap CI | `ckpt/bootstrap_ci/{phase7d_skill, ab3_era5_only, ab3b_era5_budget, ab3_cdu, ab3b_w01_sweep, ab3b_w001_sweep}/` |
| 论文图 | `ckpt/figures/{paper_per_lead, paper_skill_bar, pooled_csi, paper_case}/` |
| 训练日志 | `logs/retrain_*_20260616T2209Z.log` |
| GitHub | https://github.com/Tanhhhhtjy/Pluvian |

---

> *本子文档随 Phase 7e 重训进度更新。下次重大更新点：2026-06-17 重训完成后，§E 时间线和 §F todo 表会补新事件。*
