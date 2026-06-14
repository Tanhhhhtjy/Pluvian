# npj Climate and Atmospheric Science — Pluvian 论文叙事大纲

Created: 2026-06-14. Authoritative narrative skeleton for the in-progress submission.

## 0. 叙事结构判断

3 幕（融合 → 负结果剖析 → 架构解药）是合理结构。pooled-CSI 诊断应与 CDU 同等高度——它是可被同行复用的工具，CDU 只是一种 fix。

## 1. Abstract (draft, ~195 words)

> Short-term precipitation nowcasting over North China remains difficult due to convective intermittency and underused multimodal context. We present a 6-minute-resolution, 108-minute deep-learning nowcasting framework that progressively fuses radar with column water vapor (PWV) and ERA5 large-scale environmental fields. Stepwise fusion yields monotonic improvement across CSI at 1/5/10 mm h⁻¹, FSS, and MAE on an event-based test set, while the extreme CSI@30 mm band is preserved at the radar baseline level. Pushing further with a physically motivated water-budget loss improves light-to-moderate skill but causes extreme CSI30 to collapse from 0.291 to 0.210. Using a pooled-CSI diagnostic with neighborhood max-pool tolerance, we show this is not spatial displacement but physical smoothing of convective cores — a differentiating negative result for physics-informed loss design in nowcasting. We then introduce Cubic Dual Upsampling (CDU), a dual-branch decoder combining low-frequency bicubic interpolation with high-frequency PixelShuffle residuals, which restores extreme skill at the architecture level without loss reweighting. Contributions are threefold: a multimodal nowcasting baseline with reproducible monotonic ablations; a pooled-CSI diagnostic that separates smoothing from displacement; and an architectural prescription for preserving extreme-precipitation fidelity in physics-aware training.

## 2. 四节 outline

### §1 Introduction
- 华北短临降水的业务挑战：对流系统寿命短、组织快，6 分钟级输出对防灾响应有刚性意义
- 现有 nowcasting 以雷达外推为主，水汽场和大尺度环境驱动信息系统性缺位
- 物理约束损失流行但对极端事件的影响缺乏诊断工具，正负效应常被平均掉
- 本文三个贡献：逐级多模态融合的单调收益证据链；区分"抹平 vs 偏移"的 pooled-CSI 诊断；CDU 架构解药

### §2 Method
- 数据：华北雷达 mosaic（6 min cadence）、GNSS-PWV 插值场、ERA5 单层与气压层环境变量；event_test（事件挑选）与 test_robust（鲁棒性）两套切分
- 架构：encoder-decoder 主干 + 多模态融合通道（radar → +PWV → +ERA5），6 帧 history + 18×6 min lead 输出；CDU 双分支解码器结构定义
- 训练：损失组成（base intensity-stratified MAE + 可选水汽收支项 ab3b）、优化器、batch、epoch、curriculum
- 评测：CSI@{1,10,30} mm h⁻¹、FSS@{3,11} km、MAE；pooled-CSI 定义（max-pool 窗 1/4/16）
- **诚信声明**：单成员确定性输出，CRPS 与 MAE 数学等价（`_crps_marginal` 即 degenerate single-member），论文只报 MAE 不改名为 CRPS

### §3 Results
- 3.1 多模态融合：ab1→ab2→ab3 在 csi1/csi5/csi10/fss3/fss11/MAE 上单调提升；**诚实标注 csi30 在 ab2 阶段有 -0.018 微回归，由 ab3 ERA5 回弹至 0.291（接近 ab1 0.308）**——说明轻中量与极端的初步张力在多模态层面已经显现，但 ERA5 把张力压平
- 3.2 物理损失的极端权衡：ab3b 在 csi1/csi10/fss3 继续涨但 csi30 从 0.291 暴跌至 0.210（event_test，-28%）、0.276→0.215（test_robust，-22%），且 0/18 lead positive
- 3.3 Pooled-CSI 诊断：ab3 csi30 随邻域放宽 0.291→0.326（位移可解），ab3b 几乎不恢复 0.210→0.222（物理抹平）
- 3.4 CDU 架构解药：双分支解码器在不改损失的前提下恢复极端 skill；case study 强对流核保留对比

### §4 Discussion
- 负结果的科学含义：逐像素水汽收支残差最小化偏好平滑解，物理一致性 ≠ 极端事件保真
- pooled-CSI 作为可复用诊断：建议作为 nowcasting 标准报告项，分离"消失"与"偏移"两类失败模式
- 局限：event_test 仅 4 天事件，地理仅华北，PWV 站网密度区域差异，确定性单成员不覆盖不确定性
- 未来工作：集成多成员化使 CRPS 真实可报、扩展 dual-pol/卫星模态、跨区域迁移、物理约束改为弱形式或对抗项

## 3. 图表清单

| 编号 | 标题 | 讲什么 | 状态 |
|---|---|---|---|
| Fig 1 | 多模态融合框架与 CDU 解码器示意 | 数据流 + CDU 在网络中的位置 | **需新画** |
| Fig 2 | 四模型 per-lead 指标（CSI@1/10/30 + FSS + MAE） | ab1/ab2/ab3 单调提升 + ab3b 在 csi30 反向 | 部分已有（paper_per_lead） |
| Fig 3 | Pooled-CSI 诊断（csi30 vs pool 半径） | ab3 偏移可恢复 vs ab3b 抹平不可恢复 | ✅ ckpt/figures/pooled_csi/ |
| Fig 4 | Case study 2023-07-30T03:00 真实事件 GT/ab1/ab2/ab3/ab3b | 直观展示抹平 + 后续验证 CDU 恢复 | 在跑中 |
| Table 1 | event_test 全模型主指标汇总 | 主结论一表清 | 数据已就绪 |
| Table 2 | test_robust 同向验证 + pooled-CSI 完整数值 | 鲁棒性 + 诊断量化 | 数据已就绪 |
| Fig S1 | ERA5 通道与 PWV 输入样例 | 让审稿人看清输入信号 | 需新画（补充） |
| Fig S2 | CDU 双分支模块细节 | 架构可复现 | 可并入 Fig 1 |

## 4. 风险与缓解

| 审稿人挑战 | 应对 |
|---|---|
| CDU 不显著或不稳定 | 叙事重心退到诊断 + 负结果，CDU 降级 outlook |
| event_test 仅 4 天外推性差 | test_robust 同向 + case study 物理可解释做交叉证据 |
| 否定物理损失不普适 | 限定"逐像素 budget 残差"范畴，不否定弱形式与对抗约束 |
| 单成员不能讨论不确定性 | 明确确定性范畴 + 不改名 CRPS，集成留 future work |
| pooled-CSI 不是新工具 | 引用现有 neighborhood verification 文献，定位为"用法新"非"指标新" |
