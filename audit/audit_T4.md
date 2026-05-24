# T4 23·7 案例图 — Audit-T4 报告

- 审核人：main session（独立验证）
- 日期：2026-05-24
- 受审产出：`/Data/tanh/npj/figures/case_23p7/fig{1..6}_*.png` + `make_figs.py` + `narrative.md`
- 规范基线：T4 任务定义（DATA_HANDLING.md 第 10 节）+ npj 出版级要求

## 评级：**PASS** — Phase 3 完成，可进入论文撰写

---

## 1. 完成度

| 图 | 状态 | 文件大小 | 内容 |
|---|---|---|---|
| fig1_timeseries.png | ✓ | 448 KB | 4 子图时序 PWV/MFD/CAPE/PRE |
| fig2_pwv_jump.png | ✓ | 573 KB | PWV 3 时刻空间快照 |
| fig3_mfd_convergence.png | ✓ | 393 KB | 3 层 MFD + 风矢量 |
| fig4_cape_kindex.png | ✓ | 270 KB | CAPE + K-index 空间场 |
| fig5_trigger_visualization.png | ✓ | 1.64 MB | 4 帧雷达 + PWV 散点 |
| fig6_mechanism.png | ✓ | 394 KB | 三因子叠加复合图 |
| narrative.md | ✓ | 自补 | 每图配 100–200 字物理解读 |

**6/6 图完成 ✓**

## 2. 物理正确性（关键事件值核对）

| 图 | 期望 | 实测 | 评判 |
|---|---|---|---|
| fig1 PWV 跃升 | 30→70 mm | 30.2→69.6 mm | ✓ |
| fig1 CAPE 范围 | 0–2000 J/kg | 98–1698 J/kg | ✓ |
| fig1 降水峰值 | 数 mm/h 量级 | max 9.44 mm/h | ✓（区域均值） |
| fig2 PWV 高值带 | 36°N 以南→39°N | 视觉确认 | ✓ |
| fig3 925 hPa 辐合 | 太行山东麓 | (114–115E, 37–38N) | ✓ |
| fig4 CAPE 峰值 | >1500 J/kg | max 2155 J/kg | ✓ |
| fig4 K-index 阈值 | >38°C | max 42.7°C | ✓ |
| fig5 触发过程 | 18 BJT 弱→03 BJT 强 | 视觉确认 | ✓ |
| fig6 PWV>65 站数 | 数十站量级 | 43 站 | ✓ |

## 3. 可视化质量

- DPI: 300（出版级 ✓）
- colorbar 含 units ✓
- 子图 (a)(b)(c) 标注 ✓
- 字体可读 ✓
- 标题、坐标轴 label 完整 ✓
- 各图配色合理（PWV/CAPE 单色渐变 'YlOrRd'，MFD 双向 'RdBu_r'）

## 4. 论文承载力

- fig1 直接可作 Fig 4 候选（最强物理证据）
- fig5 直接可作 Fig 5 候选（I-5 触发预报论点视觉锚点）
- fig6 直接可作 Fig 1 候选（事件解剖图，Introduction 用）
- fig2/3/4 作 Supplementary

## 5. 小瑕疵（不影响通过）

| Issue | 影响 | 后续处理 |
|---|---|---|
| fig6 标题写"dashed contours: MFD" 但实际 MFD 等值线不明显 | 视觉小问题 | 论文成稿前可重出 fig6，把 MFD 等值线加粗或重设阈值 |
| fig1 缺 dask 第一次失败，main 补救修复后 OK | 流程小坑 | requirement.txt 加 dask 即可 |
| fig5 横向 4 panel 略压缩 | 阅读体验 | 论文成稿前可改 2x2 grid |

这些都是出版前微调级别，不阻塞当前阶段。

## 6. Phase 3 结论

**Audit-T4 PASS**。

数据 pipeline 与物理诊断量计算全流程闭环：
- T1: 数据 loader + manifest
- T2: MFD 场 15 文件
- T3: 不稳定指标 30 文件  
- T4: 23·7 案例 6 图 + narrative

**Pipeline 已具备启动 AI 模型开发的全部前置条件**。
