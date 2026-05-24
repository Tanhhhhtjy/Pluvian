# Audit-T3：不稳定指标审核报告

- 审核人：main (因团队消散，由 main session 直接审核)
- 日期：2026-05-24
- 受审产出：`/Data/tanh/npj/derived/instability/era5_{cape,cin,kindex,lcl,si,tt}_2023{05..09}.nc`（30 文件）+ `pipeline/compute_instability.py`
- 规范基线：DATA_HANDLING.md §6.2

## 评级：**PASS** — 解锁 T4

---

## 1. 时间编码（T2 同类问题复核）

- `valid_time.units = "hours since 2023-07-01 00:00:00"`（按月独立 epoch）
- `calendar = "proleptic_gregorian"` ✓ CF 标准
- 用 `cftime.num2date` 解码成功；7-28 18:00 索引 666 验证一致 ✓
- **下游使用注意**：每个月文件有独立时间 epoch（不是统一 unix），data_loader 拼接时需用 cftime 而非 raw int64

## 2. 物理量级（关键事件验证）

### CAPE — 7-28 当日逐小时 京津冀框 (36–41N, 113–119E) max
```
UTC时    CAPE J/kg    BJT      物理判读
 04:00      1765       12:00    午间增温启动
 05:00      1856       13:00    
 06:00      1909  ★    14:00    日峰 ✓ >1500 阈值通过
 07:00      1694       15:00
 08:00      1495       16:00
 18:00      1168       02:00 次日  夜间降温后偏低
```
**7-28 全天京津冀 CAPE 峰值 1909 J/kg ✓ 远超 1500 阈值**

主事件期 (7-28 → 8-1) 华北框 top-5 全部 ≥1765 J/kg，符合强对流潜势。

### 其他指标 (7-28 18:00 UTC, 36-41N/113-119E)
| 指标 | min | max | mean | 单位 | 物理判读 |
|---|---|---|---|---|---|
| K-index | 24.7 | **39.3** | 35.6 | °C | ✓ >35 暴雨阈值 |
| SI | -2.1 | 1.9 | -0.3 | °C | ✓ 负值不稳定 |
| TT | 40.6 | **46.5** | 43.3 | °C | ✓ >44 强对流阈值 |
| LCL | 800 | 925 | 893 | hPa | ✓ ~600-1700m 合理 |
| CIN | -216 | 0 | -27 | J/kg | ✓ 负值正常 |

## 3. 文件完整性

- 6 指标 × 5 月 = **30/30 文件齐全** ✓
- NaN 占比 < 1% ✓
- compute.log 显示 9 月 monthly 总 1895s, T=720 时刻全跑通

## 4. metpy 使用

- compute_instability.py 用 most_unstable_cape_cin（推荐选项，适合暴雨触发）
- pressure 维度 925→250 monotonic decreasing ✓
- 32 worker 并行（每月 ~30-40 min）

## 5. 已知 minor issues

1. **monthly time epoch 不统一**：每月有独立 epoch（"hours since 2023-MM-01"）。下游 data_loader 跨月查询时需用 cftime 解码，不能简单拼接 int64。**已是规范级注意事项**，写进 DATA_HANDLING.md。
2. **CAPE max 在 UTC 时间 06:00** (BJT 14:00)，符合日间对流峰值规律；早期审核 18UTC 取值偏低是 BJT 02:00 夜间降温所致，**不是数据问题**。

## 6. PASS 总结

| 维度 | 结论 |
|---|---|
| 时间编码 | ✓ CF 标准，cftime 可读 |
| CAPE 7-28 峰值 1909 J/kg | ✓ 通过 1500 阈值 |
| K-index 39.3 | ✓ 通过 35 阈值 |
| SI, TT, LCL, CIN | ✓ 物理量级合理 |
| 文件完整性 30/30 | ✓ |
| metpy parcel selection | ✓ most_unstable 适合暴雨触发 |

**Audit-T3 PASS**。T4 case study 可立即启动。
