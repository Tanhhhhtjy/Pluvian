# npj AI4NWP 项目候选创新点报告
日期：2026-05-24
材料：5 路并行调研（SOTA / PWV+水汽 / 物理架构 / 7.28 案例 / npj 期刊口味）

## 一、关键事实摘要（跨调研交叉验证）

### A1 — npj Climate and Atmospheric Science 的硬性偏好
- 官方 scope 明确包含"AI for weather/climate prediction"——非擦边。
- 强烈偏好「方法新颖 + 真实业务问题 + 物理一致性诊断」三件套；纯 SOTA-chasing 易被建议改投 AIES/GRL。
- 中国团队 + 中国区域数据占 35–45%，但单写"华北"会被要求与全球可比性挂钩。
- Article 类型 6000–9000 词，4–6 主图，参考文献 50–60；Brief Communication 1000–1500 词适合"短平快单点"。

### A2 — SOTA 短临降水 AI 的空白
- 2023–2025 主线是"diffusion + 单变量雷达"，CSI/CRPS 增益边际递减。
- **PWV 在 nowcasting 里几乎从未与雷达深度融合**：唯一明确用 PWV 的 GNSS 单点工作（arXiv 2509.25263, 2025）不用雷达；唯一明确做 radar+PWV+DL 的是 Liu et al. 2025 IEEE TGRS，但只做了 0–2h 单变量融合，未涉及物理约束 / 多变量 / 极端事件诊断。
- MetNet-3 是唯一系统融合"雷达+卫星+站点稀疏观测"的工作，但仍未用 PWV，且物理可解释性弱。
- 物理约束以"软引导"为主：NowcastNet 用 advection 算子作可微演化，PIANO 用 advection-diffusion 残差；**水汽收支方程 (∂PWV/∂t + ∇·(qV) = E − P) 作为 loss 至今未见**。

### A3 — PWV 文献的关键 gap
- PWV 多被当"标量预报因子"；从未有人用 GNSS 网络反演**水平 PWV 梯度场 → 代理水汽通量散度场 (MFD)**，再喂入网络作为动力学输入。
- PWV 时间特征（增长率/跳变/累积）与 CAPE/LCL 等不稳定指标的协同极少同框；He et al. 2025 IEEE JSTARS 是初步尝试但只有 SHAP 全局重要性。
- XAI 在降水预报里几乎全是 SHAP；对 PWV 这种时空场没有"反事实归因 (counterfactual attribution)"或"物理一致性检验"。

### A4 — 物理架构调研结论
- PINN 在真实雷达场不可信，只能当 regularizer (★☆☆☆☆)。
- Neural Operator 全局好、局部短临一般 (★★★☆☆)。
- **PDE soft loss** 在 nowcasting 上 CSI +3–8%，性价比最高 (★★★★☆)。
- Hybrid NN+solver (NeuralGCM 风格) 工程量过大 (★★★☆☆，仅 discussion)。
- **Diffusion + physics** 对极端段 / 概率指标最强 (★★★★☆)。
- **Multi-modal cross-attention 融合**是稀疏-稠密观测的甜区 (★★★★★)。
- 三条推荐 sketch（A=保守落地 / B=进阶 advection+diffusion / C=野心 NeuralGCM 区域版）。

### A5 — 7.28 京津冀暴雨作为旗舰案例
- "23·7"暴雨 2023-07-29 至 08-01，1883 年以来华北最大；河北临城王家垣 1003 mm。
- 触发机制共识：Doksuri 残涡 + 副高+蒙古高压双高坝 + Khanun 助推 + 太行山地形抬升。
- 业务模式（ECMWF/CMA-MESO/GFS）位置基本对，**极值低估 1/2–2/3**，过程持续时长偏短。
- Pangu/GraphCast/FengWu/Fuxi 等大模型对台风路径好，但对华北极端降水普遍存在"过平滑"问题，**至今没有针对 23·7 的 head-to-head AI 评估论文进 npj/NC**——空白。
- 物理诊断有可信材料：PWV 在 7-28 夜跃升至 70+ mm 并持续 4 天，850 hPa MFD 在太行山东麓达 −10×10⁻⁵ g/(cm²·hPa·s)，正与致死区重合。
- 同期可用作训练/对比的事件：2023-06-21/24、2023-07-15/17、2023-08-09/12、2023-08-20/22、2023-09-04/06。

---

## 二、候选创新点清单（按"novelty × 可行性 × 故事强度"打分）

每条标：**N** = novelty 强度 (1–5)；**F** = 可行性 (1–5)；**S** = 故事强度 (1–5)；**风险**。

### 创新点 I-1：PWV 收支方程作为软物理约束损失
**核心**：把 ∂PWV/∂t + ∇·(q⃗V) − E + P = 0 写成可微 loss，让网络在预报降水 P 的同时受 PWV 真观测（GNSS）与风场（ERA5/站点风）约束。
**为什么是新**：调研显示 NowcastNet 用 advection、PIANO 用 advection-diffusion，但**没人用 PWV 收支**——而 PWV 收支恰好是连接"水汽输入 (GNSS)、动力 (风场)、降水输出 (雷达/站点)"的物理桥梁，是 radar+PWV 工作的天然延续。
**N=5  F=4  S=4**
**风险**：(a) loss balancing 需要 GradNorm / 不确定度加权；(b) 离散 ∇· 算子的数值扩散；(c) E (蒸发) 难观测，可吸收进残差项。
**实施**：先 §1 工程化（构造可微 ∇·、∂t 算子），再两阶段训练（先 data 再 +physics）。
**对标**：Liu et al. 2025 IEEE TGRS 的 radar+PWV 工作（无 physics）。

### 创新点 I-2：GNSS 网络反演的 MFD 场作为显式物理输入通道
**核心**：用 30+ 个 PWV 站点的空间梯度 + ERA5 风场（或站点风内插）构造 **网格化的代理水汽通量散度场**，作为独立"物理通道"喂进网络（独立 token / 独立 branch，而非简单 concat）。
**为什么是新**：调研里 PWV 一直被当标量；An et al. 2026 (Climate, MDPI) 证明"物理特征独立分支 > 简单 concat"，但未涉及 MFD；MFD 在传统气象里被反复证明是触发指标 (Banacos & Schultz 2005)。把它做成 AI 输入通道是首次。
**N=4  F=4  S=4**
**风险**：PWV 站点稀疏（河北 216 文件 ≈ 30–60 站），梯度反演噪声需要平滑/克里金；时间分辨率 30min 与雷达 6min 错位需对齐。
**实施**：先用克里金/泊松反演 PWV 场到 0.1°，再算 ∇·；建议把 MFD 场作为 "auxiliary channel + ablation 主角"。

### 创新点 I-3：多源异构稀疏-稠密 cross-attention 融合架构
**核心**：用 Perceiver IO / MetNet-3 风格的 cross-attention，让稠密雷达 query 主动去 attend 稀疏 PWV 站点 token + 极稀疏地面气象站 token (PRS/TEM/RHU/WIN_S/WIN_D)。
**为什么是新**：MetNet-3 做了 radar+sat+sparse obs 但闭源、未公开 PWV 通道；我们做的是"radar+GNSS PWV+sfc 6-variable"组合，全球首次。
**N=3.5  F=5  S=3**
**风险**：站点位置编码 (Fourier features) + missing mask 需要写好；不算特别新的架构 idea，但是必需的工程基础。
**实施**：Swin/AFNO backbone + Perceiver-style latent + cross-attn，模板很成熟。

### 创新点 I-4：物理可解释性——PWV 反事实归因
**核心**：训练完后做"反事实实验"：人为把某区域/某时刻的 PWV 替换成气候态，看预报降水如何变化；结合时空 SHAP 与因果干预，回答"模型是不是真的学到了水汽-降水耦合"。
**为什么是新**：现有降水 XAI 几乎都是 SHAP/attention 全局重要性，无人做"时空场反事实"；对 PWV 这种"标量场"反事实有清晰物理含义。
**N=4  F=4  S=5**
**风险**：(a) 反事实需要严格 confounder 控制；(b) 工作量在分析端，不在训练端。
**实施**：单独一节 4–6 页 "Physical interpretability via PWV counterfactuals"，对 npj 审稿人极有吸引力。

### 创新点 I-5：水汽前兆 + 触发性预报 (trigger forecasting)
**核心**：把"PWV 跳变 + MFD 收敛 + CAPE 抬升"作为联合前兆信号；让模型学习"在 0 雷达回波 → 有回波"的对流触发，不只是"已有回波外推"。
**为什么是新**：调研明确指出现有 nowcasting 模型"只会外推已有回波，无法在湿度异常+辐合时无中生有地预报新对流单体"——这是公认空白。23·7 暴雨在 7-28 夜 PWV 突破 70 mm 时雷达回波尚未发展，正是触发性预报的完美 case。
**N=5  F=3  S=5**
**风险**：(a) 触发样本天然不平衡；(b) 评估指标需要新设计（不是 CSI，而是 trigger lead-time / hit rate）。
**实施**：作为论文 "故事高点"，最适合在 Discussion 里把 23·7 当 case study 强化；主实验仍以传统 CSI/FSS/CRPS 为骨。

### 创新点 I-6：23·7 暴雨作为 AI 极端事件 benchmark + 公开数据集
**核心**：把华北 2023.5–9 整理成开源数据集（radar 6min + PWV 30min + 6 站点变量 1h），并 explicit 把 23·7 当 hold-out test case，定义 5 个评估维度（CSI/FSS/CRPS/极端值偏差/触发提前时间）。
**为什么是新**：调研指出 23·7 至今没有 head-to-head AI 评估论文进 npj；FuXi-Extreme/GenCast 等"极端事件 AI"工作均以欧洲热浪/大西洋飓风为例，**华北季风极端是空白案例池**。
**N=3.5  F=5  S=5**
**风险**：低；纯整理工作。但需要确认数据可公开授权。
**实施**：建议作为论文 Data 章节的卖点 + GitHub 公开。

### 创新点 I-7（可选 ambitious）：可微 advection + diffusion 残差 (Sketch B)
**核心**：把 0–3h 主干 advection 显式写成可微 semi-Lagrangian / spectral advection 算子，diffusion 模型只负责物理之外的随机残差，借 NowcastNet+DiffCast 思想 + PWV 加权。
**N=4  F=3  S=4**
**风险**：训练分两段；推理 10–20 步去噪较慢；diffusion 集合 spread 需调。
**实施**：作为论文第二个实验（"ablation/extension"），凸显物理保真 + 概率能力。

---

## 三、推荐的论文叙事骨架

**主线（建议）**：把 I-1 + I-2 + I-3 + I-4 + I-6 组合 → 一篇 npj Article。
**故事一句话**：
> 我们提出 **PhyMet-Nowcast**（暂名）—— 首个把 GNSS-PWV 收支方程作为软物理约束、并以 PWV 网络反演的代理水汽通量散度作为显式物理通道的多源短临降水预报框架；在华北 2023.5–9 数据上相比 NowcastNet/DiffCast/MetNet-3-style baseline 在 CSI@10/20/50 mm/h 和 CRPS 上均显著优于现有 SOTA；并在 2023 "23·7" 京津冀特大暴雨上首次给出 AI 模型对该事件的物理一致诊断与提前 6–12h 的前兆预警能力。

**Figure 1**：框架图（左：多源数据流；中：cross-attn 融合 + 可微 PWV-budget 算子；右：评估 / 物理诊断）。
**Figure 2**：CSI/FSS/CRPS vs lead time 的 baseline 对比。
**Figure 3**：23·7 案例 4 张图——AI 输出 vs ECMWF vs CMA-MESO vs 实况。
**Figure 4**：PWV 反事实归因 + MFD 场叠加可视化。
**Figure 5**：消融——只 radar / +PWV concat / +PWV 独立通道 / +MFD 通道 / +PWV-budget loss 五档。
**Figure 6**：触发性预报 case（7-28 夜雷达回波从无到有）。

**备选投稿**：若 npj 反馈"偏方法"，转 **AIES** 或 **GRL**（GRL 适合压缩成 4 主图的 short 版本）。

---

## 四、风险评估与备选路径

| 风险 | 缓解 |
|---|---|
| PWV 站点稀疏导致 MFD 反演噪声大 | 用克里金 + ERA5 风场约束；做敏感性分析 |
| 物理 loss 数值扩散 | 用 semi-Lagrangian + flux limiter；两阶段训练 |
| 23·7 单一案例无法支撑泛化结论 | 同期 5 次中等事件做 hold-out validation pool |
| diffusion 集合训练成本 | I-7 可以推迟到 v2 / 期刊修回阶段加入 |
| 期刊偏好不匹配 | AIES/GRL 备选，叙事内核（PWV+物理）不变 |

---

## 五、下一步建议优先级

1. **先决策**：用户确认主线（建议 I-1+I-2+I-3+I-4+I-6），决定是否包含 I-5/I-7。
2. **小规模可行性**：用 1 个月 (2023-07) 数据跑 baseline (NowcastNet 复现 + 我们的多源 cross-attn)，验证 PWV 通道是否带来增益。
3. **MFD 反演 pipeline**：先把 30min PWV 站点数据转成 0.1° 格点场，验证反演精度。
4. **可微 PWV-budget 算子**：写一个 PyTorch 模块（∇·, ∂t），单元测试通过后接入。
5. **23·7 案例**：作为 hold-out，主实验完成后再跑专项分析。
