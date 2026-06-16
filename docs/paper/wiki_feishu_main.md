# Pluvian — 多模态深度学习短临降水预报

> 项目协作主文档 | 投稿目标：*npj Climate and Atmospheric Science* | 仓库：[Tanhhhhtjy/Pluvian](https://github.com/Tanhhhhtjy/Pluvian)
>
> 📂 **细节请参阅子文档**：*Pluvian — 实验过程与决策记录*（process 子页）

---

## 0. Abstract

> Short-term precipitation nowcasting over North China remains difficult due to convective intermittency and underused multimodal context. We present a 6-minute-resolution, 108-minute deep-learning nowcasting framework that progressively fuses radar with column water vapour (PWV) and ERA5 large-scale environmental fields. Stepwise fusion yields monotonic improvement across CSI at 1/5/10 mm h⁻¹, FSS, and MAE on an event-based test set; CSI@30 is non-monotonic across the cascade. Pushing further with a physically motivated water-budget loss improves light-to-moderate skill but causes extreme CSI@30 to collapse. Using a pooled-CSI diagnostic with neighbourhood max-pool tolerance, we show this is not spatial displacement but physical smoothing of convective cores — a differentiating negative result for physics-informed loss design. We further introduce Cubic Dual Upsampling (CDU); CDU avoids the budget-loss trade-off without itself recovering the extreme tail, which we frame as an open architectural problem.

---

## 1. 项目概览

**研究目标**：构建能利用多模态物理观测（雷达 + GNSS-PWV + ERA5[包括风速、湿度等物理量]）的深度学习短临降水预报框架，重点关注极端降水事件的预报。

**数据集**：

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

## 3. 方法论

### 3.1 模型架构总览

整体是 **encoder–decoder** 主干 + **多模态融合** 的设计。三条信息流在 bottleneck 处会合：

- **雷达分支（dense grid）**：标准 conv stem + 两层 stride-2 下采样到 H/8 × W/8，叠两层 windowed self-attention 块（捕获空间局部依赖）+ 一层 temporal block（捕获 12 帧时序演化）。中间三层 feature map（H/2, H/4, H/8）作为 U-Net skip 出来给 decoder。
- **PWV 分支（sparse token）**：GNSS-PWV 站点不在规则网格上，所以不用 conv，而是把每个站的 (PWV value, lon, lat, time) 编码成 token，给 Fourier positional encoding，进 cross-attention 到 radar 的 dense feature 上。这个设计避免把稀疏观测插值上栅格再卷积所引入的过度平滑。
- **ERA5 分支（coarse grid）**：ERA5 是 0.25° 网格的 reanalysis，u/v/q/T 四个变量在 8 个 pressure level 上展开成 32 channels。先用一个轻量 conv tower 在 coarse grid 上算 feature，再 bilinear 上采样并残差加到 H/8 的雷达 feature 上。**重点：ERA5 的重计算保留在 coarse grid 上**，这是 mismatch-aware 的设计，避免直接 1 km 上跑 32-channel attention 爆显存。

三个分支在 bottleneck 处由 **GatedAsymmFusion** 模块融合：一个 learnable gate 在像素层面决定 PWV 信号 vs 站点观测信号哪个权重更高，避免把这两条本来语义不同的稀疏通道粗暴 concat。

decoder 阶段先用一个 **per-spatial-location cross-attention** 把 12 帧的融合 token 映射到 18 个 learnable lead-time queries —— 每个未来帧"自主选取"它最关心的历史片段。然后通过三阶 U-Net 上采样（H/8 → H/4 → H/2 → H）回到原分辨率，每阶都有 skip-fuse。最后两个 head：一个 single-channel rain head（输出 mm/h），一个 single-channel PWV head（用于辅助物理一致性损失）。

### 3.2 三个核心创新

**(i) 对稀疏 / 网格 / 物理量的非对称融合**

我们没有把 PWV 和 ERA5 都强行插值到雷达网格上再 channel-concat，而是给每条信息流一个跟它"自然语义"匹配的接口：
- PWV 留在 sparse token 域，靠 cross-attention 进入 dense radar
- ERA5 留在 coarse grid，靠残差 add 进入 H/8 dense

这背后的判断：模态对齐（modality alignment）的关键不是分辨率统一，而是**让模型自己学跨模态关联**。

**(ii) 物理 PDE 约束作为辅助损失（探索性）**

我们尝试把水汽收支方程作为 loss term：

$$\varepsilon = \partial \text{PWV} / \partial t + (\Delta p / g) \cdot \nabla \cdot (qV) + P_{\text{predicted}}$$

理论上要求模型预测的降水 $P$ 和环境水汽通量散度物理一致。这是 PINN-style 的物理约束，文献里近年很流行。**但我们的实验揭示这种 pointwise PDE 约束有副作用** —— 这正是论文第二幕（negative result）和 pooled-CSI 诊断工具（贡献 2）出现的动机。

**(iii) 双分支解码器 (CDU) 替代单一上采样路径**

标准 PixelShuffle decoder 在做 ×2 上采样时会隐式低通滤波——高频细节（强对流核的尖锐边界）容易被平均掉。CDU 把上采样拆成两条独立路径：低频分支显式保留平滑场，高频分支专门学**残差**纹理，最后 concat 融合。这是从频域思维做的架构修改，跟损失函数完全正交。

### 3.3 训练协议

| 维度 | 取值 |
|---|---|
| 损失主项 | intensity-stratified loss（5 个强度桶 × 长尾权重 × CE + weighted MSE） |
| 辅助损失 | FSS-based neighbourhood loss（多阈值，weight 0.1） |
| 探索性 | water-budget PDE residual loss（仅 ab3b 用，weight 0.1） |
| Optimizer | AdamW，lr 3e-4，cosine + 2 epoch warmup |
| Batch | 物理 1 + grad-accum 4 = effective 4，bf16 mixed precision |
| Epoch | 60 |
| 硬件 | RTX 4090 24 GB / 模型，~1100 s / epoch |
| 评测 | full-set CSI（累加 hits/FA/miss 后算）、FSS、MAE、paired bootstrap CI |

**关键设计选择**：
- 所有 ablation（ab1/ab2/ab3/ab3b/CDU）共享**同一份训练 schedule、loss 配置、随机种子和评测协议**。这样模型间的差异**只来自 input modality 或 architecture 的单一改动**，否则 ablation 结论会被超参数差异污染。
- intensity-stratified loss 是为了对抗"长尾分布塌陷"——降水分布极度长尾，朴素 MSE 会让模型预测大面积常规小雨 + 完全 miss 极端尾部。我们用强度桶的样本权重 [0.1, 1, 2, 4, 8] 显式让 loss 对高强度 band 更敏感。

### 3.4 评测协议

- **Test split 分两类**：`event_test` 是 4 个 storm 天（极端事件评测），`test_robust` 是 8 个普通天（鲁棒性评测）。这两个 split 对应的 narrative 不同：前者答"对极端事件预报得怎么样"，后者答"在日常普通天气下还稳不稳"。
- **统计严谨性**：所有 CSI/FSS 都按 full-set 累加 hits/FA/miss 再算（避免 batch-mean 对小批样本敏感）；ablation 之间的差异用 **paired bootstrap delta CI**（n_boot = 2000，按天分层）验证统计显著性，而不是只看点估计。
- **Pooled-CSI 诊断**：核心方法学贡献。同一份预报和真值，分别在 max-pool window w ∈ {1, 4, 16} 像素下计算 CSI——通过比较 CSI 随 w 的恢复曲线，把 displacement 错误 vs smoothing 错误分开。这把传统 CSI / FSS 都聚合掉的两类失败模式显式拆出来。

---

## 4. 下一步与风险

### 4.1 路线图

**Phase 7e**：所有 5 个模型在重训中（mm/h 单位修正后从头训），完成后会用新数据重出全部论文图、表、CI。这是论文成稿前的 critical path。

**Phase B — SOTA 对比**：在我们的数据 / 切分 / 评测协议下重训 4 个公开 baseline（pysteps STEPS / Earthformer / exPreCast / DGMR），radar-only + 手工加 ERA5 channel 两套，跟 ab1 和 ab3 分别公平对比。这是 reviewer 必问的"你们和已发表方法比怎么样"。

**投稿前**：
- 架构示意图（Fig 1）+ 输入示例图（Fig S1）
- Bibliography 从 placeholder 转 BibTeX
- §1–§4 数字最终 audit + reviewer 回信

### 4.2 已识别的风险与限制

| 限制 | 影响 | 已采取的应对 |
|---|---|---|
| event_test 仅 4 storm 天 / test_robust 仅 8 天 | 外推性受限，reviewer 必问"能否泛化到其他季节、其他区域" | 论文 §4.5 显式 scope；多季节 / 多区域留 future work |
| 单成员确定性输出 | 无 ensemble spread，不能讨论概率预报 | 只报 MAE 不报 CRPS；集成留 future work |
| 训练数据仅 2023 年 5–8 月 | 冬季层状 / 混合相态降水未覆盖 | limitation 显式标注 |
| 公开 multimodal nowcasting SOTA 代码缺失 | FusionCast / MAG-Net / VMU-Diff 全闭源，做不到 fair multimodal SOTA 对比 | 在 Phase B 中只能给开源 radar-only baseline 手工加 ERA5 通道，作为 our extension 比较 |
| 物理 PDE loss 的负结果在文献里没有先例 | §4.2 论点要靠我们自己的 ablation 证据撑 | 通过 budget weight sweep 增强论点严谨性 |

详细的过程数据、reviewer concerns 完整对策、决策时间线见 process 子文档。
