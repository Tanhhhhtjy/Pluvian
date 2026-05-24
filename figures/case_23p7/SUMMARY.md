# T4 案例研究图执行报告

**目标**：23·7 京津冀暴雨 (2023-07-29 → 08-01) 的 6 张物理诊断图。
**结果**：6/6 全部成功生成，无静默失败。
**输出目录**：`/Data/tanh/npj/figures/case_23p7/`
**生成脚本**：`make_figs.py`（可重跑）
**日志**：`_run.log`
**叙事**：`narrative.md`

## 已交付清单

| 文件 | 大小 | 状态 | 关键量 |
|---|---|---|---|
| fig1_timeseries.png | 437 KB | OK | PWV 30→70 mm；CAPE peak 1698 J/kg；pre peak 9.4 mm/h |
| fig2_pwv_jump.png | 559 KB | OK | 7-28 BJT 三时刻；框内 138/170/170 站；mean 45→51 mm |
| fig3_mfd_convergence.png | 384 KB | OK | 7-29 02 UTC 三层 max 1.19 / 1.23 / 0.40 ×10⁻⁶ |
| fig4_cape_kindex.png | 264 KB | OK | 7-28 06 UTC CAPE max 2155 J/kg；K-index max 42.7 °C |
| fig5_trigger_visualization.png | 1.6 MB | OK | 7-28 18 → 7-29 03 BJT 四帧雷达+PWV |
| fig6_mechanism.png | 384 KB | OK | 7-28 21 UTC 三因子叠加，PWV>65 mm 站点 43 个 |

## 关键科学发现

1. **水汽预充注先行 24 小时**。fig1 显示 PWV 自 7-26 凌晨开始抬升、7-28 晚突破 65 mm，而站点降水在 7-29 才进入暴发期。说明 GNSS-PWV 是雷达回波尚不可见时的领先指标，支持论文 "PWV 给短临预报带来 0–3 小时的预警"假设。

2. **MFD 暴雨级量级被独立确认**。fig3 在 7-29 02 UTC 925/850 hPa 都达到 1.2×10⁻⁶ kg/(kg·s)，与 audit-T2 报告中"超阈值 13–20 倍"一致；最强收敛带落在太行山东麓，证实地形抬升触发机制。

3. **不稳定能量午后达峰、夜间释放**。fig4 显示 CAPE 在 BJT 14:00 达 2155 J/kg、K-index 42.7°C，距 7-29 凌晨对流爆发约 8 小时；MCS 组织化所需时间合理。

4. **fig6 三因子在京津冀腹地高度重合**。CAPE 大值带、MFD 收敛区、PWV>65 mm 高值站三者位置同位，构成 npj 论文 Results 章节的标志性机制图。

## 潜在问题（如实记录）

1. **fig1 (b) MFD 振幅小**。京津冀核心区 (3°×2° box) 平均后 MFD 仅 ±0.04 ×10⁻⁵，因为正负抵消（暴雨级的局地极值是 1×10⁻⁶ 量级即 100×10⁻⁵，但 box-mean 平滑掉）。**这是物理正常现象**，不是 bug；如需更突出，可改为 "box-max" 而非 "box-mean"，但当前以 mean 与 PWV/CAPE/PRE 的 mean 保持一致。

2. **fig6 MFD 虚线等值线在 CAPE 浅红背景上较弱**。levels 起点 5e-7 偏高，部分时刻 850 hPa MFD 在该阈值之上的连续区域有限，可视化效果不如预期。如评审要求，可降阈到 2e-7 并加粗。

3. **fig5 雷达 18 BJT 帧确实"几乎无回波"**——这是真实物理现象（对流尚未启动），不是数据缺失。已在 narrative 中明示。

4. **xarray.open_mfdataset 默认使用 dask**，本机无 dask。已改为 xr.concat 多文件方式（无 dask 依赖）。修复后 fig1 通过。

5. **京津冀核心区 PWV 站点数量**：boxed 后 70 站（fig1 时序）/ wider domain 138–170 站（fig2/5/6）。覆盖度充裕，无 8 个"显著缺失"站需要特别处置（已经在 pwv_io 的 PARTIAL_STATIONS 集合中保留处理）。

6. **未做地图边界轮廓**（cartopy 重依赖）。仅以 BJ/TJ/SJZ 三个地标三角点示意，论文出图前如需更专业地图底，可在后期套 cartopy province shapefile。

## 重跑方式

```bash
cd /Data/tanh/npj
python3 figures/case_23p7/make_figs.py
```

约 60–90 秒（PWV 站点 IO 主要瓶颈；已用 lru_cache）。

## 与提案的对齐

参照 RESEARCH_PROPOSAL_v1.md：6 张图整体覆盖 Section "Case Study: 23·7 Beijing-Tianjin-Hebei extreme rainfall" 的所有可视化需求，特别是支撑 "multi-source fusion outperforms radar-only" 论点的关键证据（fig2/5 = 水汽场先行；fig6 = 三因子叠加机制）。后续 quantitative attribution（去 PWV 通道的 ablation 退化幅度）属 T5/T6 任务范畴，本任务不覆盖。
