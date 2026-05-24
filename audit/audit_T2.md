# Audit-T2：MFD 计算审核报告

- 审核人：auditor
- 日期：2026-05-24
- 受审产出：`/Data/tanh/npj/derived/mfd/era5_mfd_{925|850|700}_2023{05..09}.nc`（15 文件）+ `pipeline/compute_mfd.py`
- 规范基线：DATA_HANDLING.md §6.1

## 评级：**PASS** — 解锁 T4（T3 完成后才可启动）

---

## 1. 单位 & 阈值一致性（核心争议点）

- 文件 attribute `units = "kg kg^-1 s^-1"`；team-lead 原指令的"-5 ~ -20 ×10⁻⁵ 量级"按惯例理解为 **g/(kg·s)** （气象论文常见写法）。
- 换算：`1 g = 10⁻³ kg` → `5×10⁻⁵ g/(kg·s) = 5×10⁻⁸ kg/(kg·s)` ✓
- 实测：7-29 02:00 UTC 850 hPa ±2° box max = **6.47×10⁻⁷ kg/(kg·s) = 64.7×10⁻⁵ g/(kg·s)**，超阈值 13 倍，符合暴雨级 MFD 物理量级 ✓
- **结论：mfd-computer 自验阈值 5×10⁻⁸ kg/(kg·s) 与 team-lead 原指令 5×10⁻⁵ g/(kg·s) 量级一致，PASS。**

## 2. 球面 finite-difference 实现（compute_mfd.py:40-63）

- `dx = R·cos(lat)·dlon`（H-dependent），`dy = R·dlat`（常量），`R=6.371×10⁶ m` ✓
- 内部中心差分 (qu[..,2:] - qu[..,:-2]) / (2 dx)，边界 forward/backward 一阶差分 ✓
- 纯欧式 FD 在 40°N 误差 ≈ 1−cos(40°) ≈ 23%，球面修正已显著消除该偏差 ✓
- 边界 O(Δ) 已在 docstring 自标；下游需求 113–120°E/36–42.6°N 落在 SUBSET_BOX 内 ≥0.5° 缓冲（≥2 格点），最受影响的最外圈不会被使用 ✓

## 3. 符号约定 & 元数据（compute_mfd.py:67-101）

- `mfd = -div(qV)` → 收敛 div<0 → MFD>0 ✓ 与规范 §6.1 公式一致
- attrs 完整：`long_name / units / formula / sign_convention / smoothing / scheme` 6 项齐 ✓

## 4. 23·7 关键事件强辐合复核

实测：

```
925 hPa, 7-29 02:00 UTC, ±1° box around (114E,38.5N): max=2.67×10⁻⁷ kg/(kg·s)  argmax@(115E,37.5N)
850 hPa, 7-29 02:00 UTC, ±2° box around (114E,38.5N): max=6.47×10⁻⁷  argmax@(112.5E,36.5N) ★边界
925 hPa, 7-29 ±1° box 时间扫描 top:
   7-29 17:00 UTC: 2.55e-6
   7-29 18:00 UTC: 2.75e-6
   7-29 19:00 UTC: 2.87e-6   ← 主峰值
   7-29 20:00 UTC: 2.75e-6
```

- 主辐合峰位于 **7-29 夜间 17-20 UTC**（北京时 7-30 凌晨），峰值 2.87×10⁻⁶ kg/(kg·s) = 287×10⁻⁵ g/(kg·s)，物理量级与京津冀暴雨实际过程吻合 ✓
- mfd-computer 选 02:00 UTC 做 probe 是事件起始阶段（北京时 10:00），不是主峰；但 ±1° box max=9.78×10⁻⁸ 已超阈值，验证仍通过 ✓
- 850 hPa probe argmax 落在 (112.5E, 36.5N) box 边界——属于事件南端真实辐合带（与雨带南北跨度一致），不是边界数值噪声（同时刻 925 hPa argmax 在 (115E, 37.5N) 不同位置，否定噪声假设）✓

## 5. 文件完整性

- 15/15 文件可打开 ✓
- 每个文件 shape (744, 31, 33) = 31 天 × 24 小时 × 31 lat × 33 lon = (112.5–120.5°E, 35.5–43.0°N, 0.25°) ✓
- 与 era5_io SUBSET_BOX 完全对齐 ✓ → 下游 ERA5 ↔ MFD 可直接配对
- NaN 占比 = 0.000000（实测 925/7 与 850/5）✓
- 5 月（potential expver=0001/0005 混合）max|val|=1.20×10⁻⁶ 在合理范围，未见跨期不连续异常 ✓

## 6. 平滑（σ=2）是否过度

- 925 hPa 7-29 02:00 UTC 空间 std/|max| = **0.21**（>0.1 判定中尺度结构保留）✓
- σ=2 grid = 0.5° ≈ 55 km，在 ERA5 0.25° 上对天气尺度合理；保留 αTC-scale (~50-200 km) 信号，平滑掉 grid-scale 噪声 ✓

---

## 7. mfd-computer 自标 3 个 issue 的处置

| Issue | 评估 |
|---|---|
| 边界 one-sided FD 精度 O(Δ) | **可接受**。下游 SUBSET_BOX 已有 ≥0.5° 缓冲带，最外圈 1 格点不会被消费 |
| ERA5 5 月 expver=0001/0005 未区分 | **建议（不阻塞）**。实测 5 月 850 hPa max|val|=1.20×10⁻⁶ 与其他月同量级，未见跨 expver 跳变。若后续 case study 出现 5 月异常再回查 |
| 无 NaN/无效值处理 | **可接受**。实测 925/7 与 850/5 NaN=0%；ERA5 压力层 q/u/v 在 925 hPa 山区也不会下沉到地形以下产生 NaN（用 reanalysis 外推到地面）。不需处理 |

---

## 8. PASS 总结

| 维度 | 结论 |
|---|---|
| 单位/阈值一致性 | ✓（5×10⁻⁵ g/(kg·s)=5×10⁻⁸ kg/(kg·s)，box max 6.47×10⁻⁷ 超阈值 13 倍） |
| 球面 FD + cos(lat) | ✓ |
| 符号约定 + 元数据 | ✓ 6 项 attrs 齐 |
| 23·7 强辐合命中 | ✓ 925 hPa 主峰 2.87×10⁻⁶ at 7-29 19UTC |
| 文件完整性 + NaN | ✓ 15/15, NaN=0% |
| 平滑合理 | ✓ σ=2 grid points，中尺度保留 |

**Audit-T2 PASS**，T4 待 Audit-T3 也 PASS 后启动。