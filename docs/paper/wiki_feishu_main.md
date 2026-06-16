# Pluvian — 多模态深度学习短临降水预报

> 项目协作主文档 | 投稿目标：*npj Climate and Atmospheric Science* | 仓库：[Tanhhhhtjy/Pluvian](https://github.com/Tanhhhhtjy/Pluvian) | 当前分支：`phase7b/integrated` | 最新 commit：`316f1b9` | 文档版本：2026-06-17 v1
>
> 📂 **细节请参阅子文档**：*Pluvian — 实验过程与决策记录*（process 子页）

---

## 0. Abstract

> Short-term precipitation nowcasting over North China remains difficult due to convective intermittency and underused multimodal context. We present a 6-minute-resolution, 108-minute deep-learning nowcasting framework that progressively fuses radar with column water vapour (PWV) and ERA5 large-scale environmental fields. Stepwise fusion yields monotonic improvement across CSI at 1/5/10 mm h⁻¹, FSS, and MAE on an event-based test set; CSI@30 is non-monotonic across the cascade. Pushing further with a physically motivated water-budget loss improves light-to-moderate skill but causes extreme CSI@30 to collapse. Using a pooled-CSI diagnostic with neighbourhood max-pool tolerance, we show this is not spatial displacement but physical smoothing of convective cores — a differentiating negative result for physics-informed loss design. We further introduce Cubic Dual Upsampling (CDU); CDU avoids the budget-loss trade-off without itself recovering the extreme tail, which we frame as an open architectural problem.

> ⚠️ **当前数据状态**：4.x 节的具体指标基于旧 dBZ-era 数据。2026-06-16 发现 dataset 单位事故，5 路并行重训进行中（mm/h 单位），ETA 2026-06-17 上午。新数字预期收紧但故事方向不变。详见 process 子文档 §A。

---

## 1. 项目概览

**研究目标**：构建能利用**多模态物理观测**（雷达 + GNSS-PWV + ERA5）的深度学习短临降水预报框架，重点关注**极端降水事件**的预报技能。

**论文三幕叙事**：

| 幕 | 论点 | 性质 |
|---|---|---|
| 一 | 多模态融合（radar→+PWV→+ERA5）单调提升轻中量预报技能 | 正面贡献 |
| 二 | 物理 PDE 损失（water-budget）能继续推高 bulk 指标，但**抹平**极端对流核 | 诚实负结果 |
| 三 | CDU 双分支解码器在架构层 sidesteps 物理损失副作用，但极端 tail 恢复仍是 open problem | 架构 ablation |

**当前阶段**：论文初稿（§1–§4 + Tables 1–2）已经过两轮独立同行评审，正在解决 reviewer 提出的 unit-consistency 问题（Phase 7e 重训中）和 SOTA baseline 缺失（Phase B 计划中）。

**数据集摘要**：

| 维度 | 取值 |
|---|---|
| 雷达域 | 华北 661 × 701，0.01° ≈ 1 km，36–42.6°N，113–120°E |
| 时间分辨率 | 6 min |
| 输入窗口 | 12 帧历史 (72 min) → 18 帧预报 (108 min) |
| 训练 / 测试 split | train 74 天 / val 8 天 / test_robust 8 天 / event_test 4 storm 天 |
| 时间范围 | 2023 年 5–8 月（PWV 覆盖期） |
| Multimodal 输入 | radar dense grid + GNSS-PWV sparse tokens + ERA5 32 channels |

---

## 2. 核心贡献

### 贡献 1 — 可复现的多模态融合 ablation

构造三个共享 backbone / schedule / loss 的嵌套配置：
- **ab1 (radar)** — 雷达单模态 baseline
- **ab2 (+PWV)** — 加 GNSS-PWV cross-attention 通道
- **ab3 (+ERA5)** — 再叠加 ERA5 stack（u/v/q/T × 8 hPa levels）

cascade 在 CSI@1/5/10、FSS、MAE 上**单调提升**；CSI@30 在 ab2 阶段微回归后由 ERA5 拉回近 ab1 水平 — **honest 框定为非单调**，foreshadow 第二幕的 trade-off。

### 贡献 2 — Pooled-CSI 诊断工具

我们提出**单参数**扩展的 fuzzy verification 用法：对预报和真值都做 max-pool（窗 w ∈ {1, 4, 16} px）后再算 CSI。

诊断逻辑：
- **Displacement (位置错)**：pooled CSI 应**随 w 增大而恢复**
- **Smoothing (强核抹平)**：pooled CSI 不论 w 多大都**不恢复**

把 CSI 和 FSS 都聚合掉的失败模式分离出来，是论文最有原创性的方法学贡献之一，定位为 "a diagnostic use of established neighbourhood verification techniques [Ebert 2008; Roberts & Lean 2008; Mittermaier 2014]"。

### 贡献 3 — CDU 双分支解码器（架构 ablation）

借鉴 exPreCast (Song et al., ICLR 2026) 的 **Cubic Dual Upsampling**：低频分支（bicubic + 1×1 conv）+ 高频分支（PixelShuffle ×2 残差），concat 后 3×3 conv 融合。仅替换 `_UpBlock` 为 `_CDUUpBlock`，其他完全一致。

**结果**（dBZ-era 数据，pending re-train）：CDU 在 bulk 指标上**全面优于** ab3，但 CSI@30 与 ab3 essentially 持平 — **CDU avoids the trade-off but does not recover the extreme tail**。这是 honest 的论文叙事：架构 ablation 证明 trade-off 不是 bulk-skill 提升的必然代价，但极端恢复仍是 open architectural problem。

---

## 3. 关键实验结果

> 以下为 dBZ-era 数据，pending mm/h 单位下重训。

### 3.1 主指标天梯（event_test, 96 windows）

| Model | Modality | CSI@1 | CSI@10 | CSI@30 | MAE |
|---|---|---:|---:|---:|---:|
| ab1 | radar | 0.397 | 0.565 | **0.308** | 5.19 |
| ab2 | +PWV | 0.432 | 0.591 | 0.283 | 4.83 |
| ab3 | +ERA5 | 0.470 | 0.632 | 0.291 | 4.27 |
| ab3b | +budget loss | 0.492 | 0.641 | **0.210** ↓ | 4.10 |
| ab3-cdu | +CDU decoder | **0.488** | **0.645** | 0.285 | **0.08** |

**关键观察**：
- ab1→ab2→ab3：bulk 单调上升，CSI@30 在 ab2 微回归
- ab3b：bulk 全场最优，**但 CSI@30 暴跌 −28%**（paired bootstrap CI 严格 < 0）
- ab3-cdu：bulk 改进，CSI@30 持平 — sidesteps trade-off，未恢复 tail

### 3.2 Pooled-CSI 诊断 — 核心方法学贡献

CSI@30 在 max-pool window w ∈ {1, 4, 16} px 下的演化（event_test）：

| Model | w=1 | w=16 | 解读 |
|---|---:|---:|---|
| ab3 | 0.291 | **0.326** | Displaced — 邻域容忍下回升 |
| ab3b | 0.210 | 0.222 | **Smoothed** — 不论邻域多大都救不回 |

→ ab3b 的失败模式锁定为 "强对流核被物理性抹平"，这是论文最有 differentiating power 的实验证据。

---

## 4. 进度与状态

### 4.1 论文章节状态

| 章节 | 状态 |
|---|---|
| §1 Introduction | ✅ 初稿 ~620 字 |
| §2 Method | ✅ 初稿 ~1500 字 |
| §3 Results (4 subsections) | ✅ 初稿，需 Phase 7e 后数字替换 |
| §4 Discussion | ✅ 初稿 ~880 字 |
| Tables 1/2 | ✅ 含 bootstrap CI |
| Fig 1（架构示意） | ❌ 待绘 |
| Fig S1（输入示例） | ❌ 待绘 |
| Bibliography | ❌ placeholder → 真 BibTeX |

### 4.2 同行评审

- **Round 1**（2026-06-15）：8 个 concerns。**C1（CI 重叠）已用 paired bootstrap 反驳**；C4/C7/C8/M1/M2 已修订
- **Round 2**（2026-06-16）：3 个新 concerns（N1 cherry-picking / N2 §3.4 vs Fig 4 不自洽 / N3 framing 弱化）

详细 reviewer 报告与对策见 process 子文档 §C。

### 4.3 当前正在进行

🔄 **Phase 7e 5 路并行重训**（mm/h 单位）：

| GPU | 模型 |
|---|---|
| 0 | ab1 (radar) |
| 3 | ab2 (+PWV) |
| 4 | ab3 (+ERA5) |
| 5 | ab3b (+budget loss) |
| 6 | ab3-cdu (**from-scratch**, 自动解决 reviewer C6) |

ETA：**2026-06-17 上午**（5 个新 ckpt 全完）。

---

## 5. 下一步计划

### 5.1 Phase 7e 收尾（明天）

重训完成后自动触发：5 模型 holdout eval → bootstrap CI → paired delta → pooled-CSI → 论文图重出 → §1–§4 数字全替换 → **第三轮 reviewer**。

### 5.2 Phase B — SOTA Baseline（约 2-3 周）

Reviewer round-1 C3 要求加外部 baseline 对比。已确认 Tier 1+2 方案：
- pysteps STEPS（统计基线）
- Earthformer (NeurIPS 2022)
- exPreCast (ICLR 2026)
- DGMR (Nature 2021)
- 给上述 baseline 手工加 ERA5 channel 作为 multimodal extension

详细 baseline 调研与执行计划见 process 子文档 §D。

### 5.3 投稿前 todo

| 项 | 估时 |
|---|---|
| Fig 1 / Fig S1 | 1 天（导师拍板风格） |
| Bibliography → BibTeX | 0.5 天 |
| 论文最终 polish + reviewer 回信 | 1 天 |

---

## 6. 风险与限制

| 限制 | 影响 |
|---|---|
| event_test 仅 4 storm 天 | 外推性受限，reviewer 必问 |
| 单成员确定性输出 | MAE/CRPS 数学等价，不能讨论 ensemble spread |
| 训练数据仅 2023 年 5–8 月 | 冬季层状 / 混合相态未验证 |
| 公开 multimodal nowcasting SOTA 代码不足 | FusionCast / MAG-Net / VMU-Diff 全闭源，无法 fair retraining |

均已在论文 §4.5 显式声明。

---

## 7. 资源链接

- **GitHub 仓库**：https://github.com/Tanhhhhtjy/Pluvian
- **本机工作目录**：`/data4/WuMingrui/TianJinyu/npj/pluvian`
- **完整决策记录**：`docs/experiments/pluvian_phase7d_ledger.md`
- **论文初稿**：`docs/paper/{MAIN.md, section_1-4, tables_1_2}`
- **Reviewer reports**：`docs/paper/REVIEW_round{1,2}.md`
- **过程子文档**：*Pluvian — 实验过程与决策记录*（飞书 wiki child page）

---

> *本主文档为高层汇报。所有过程细节、踩坑、决策推理、子实验数据见 process 子文档。所有指标在论文最终发表前可能调整。*
