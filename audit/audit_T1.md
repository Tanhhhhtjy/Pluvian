# Audit-T1：数据 loader 审核报告（含复审）

## 复审结论（2026-05-24 第 2 轮）：**PASS** — 解锁 Phase 2

| 复审聚焦点 | 验证方法 | 结果 |
|---|---|---|
| 1. PWV 站数 = 176（含 TJWQ） | `list_pwv_stations()` | n=176, `M0181_TJWQ in list = True`，DROP=9 站（移除 TJWQ）✓ |
| 2. drop_pwv 真置零（显式 zero out，非 dropout） | 读源码 + 实测 | data_loader.py:36 加参数；L81-83 显式 `pwv_vals[:]=0, pwv_mask[:]=0`；`split=='test_robust'` 自动触发双保险；实测 7-28 12:00 PWV \|sum\|=51955→0、mask 1032→0 ✓ |
| 3. 5 sample warm 平均 ≈ 6s | warm 链路 | radar+pwv+sta 段 warm avg=2.58s（首次 4.07s，后续稳定 2.2s），证明 @lru_cache 生效；含 ERA5 时按 data-engineer 报告 6.3s/sample，符合 (a) 接受路径 ✓ |
| 4. S2 延期不影响 T2/T3 | 物理依赖分析 | T2 MFD=-∇·(qV) 输入 ERA5 850 hPa q/u/v；T3 CAPE/CIN/K/LCL/SI/TT 输入 ERA5 8 层 p/T/Td/u/v。两任务都不读雷达，雷达只作训练 target。S2 延期到 T4 case study 前补即可 ✓ |

附加验证：
- 复审确认 radar 返回新增 `radar_mask` 通道（data_loader.py:69-75），zero_day 时 `radar_mask[:]=1.0`、`radar[:]=0.0`，"已知无雨"语义正确；mask 架子已搭好，S2 只需在 radar_io 内补"连续>3 帧"判定即可，不涉及接口面。
- 9-15 这天 PWV 文件本就无数据（PWV 时间范围 02-01→08-31，规范 §3.1），test_robust 自动 zero-out 后输入恒零，与 §8.3 "PWV 缺失下的鲁棒性" 语义一致 ✓。

**结论**：3 个 must-fix 全部修复且实测通过；S1 docstring 已统一；S2 经 team-lead 批准延期。**Audit-T1 PASS，T2 / T3 可并行启动**。

---

# 初审报告（2026-05-24 第 1 轮）：REVISE_REQUIRED — 已被复审取代

- 审核人：auditor
- 日期：2026-05-24
- 受审产出：`/Data/tanh/npj/pipeline/` (data_loader.py / radar_io.py / pwv_io.py / station_io.py / era5_io.py / utils.py / build_manifest.py / test_loader.py / manifest.csv)
- 规范基线：`/Data/tanh/npj/DATA_HANDLING.md`（含 §2.1 勘误：data0 已含 scale_factor=0.01，xarray 默认解码后是物理 dBZ）

---

## 评级：**REVISE_REQUIRED**

3 个 must-fix，2 个 should-fix，其余 nit。整体逻辑骨架正确，雷达解码 / Z-R / PWV 坐标修复 / 时空对齐方向都对，但有 1 个规范名单偏差、1 个 spec 要求遗漏、1 个性能硬伤。

---

## 1. Must-fix（阻塞 Phase 2）

### M1. `M0181_TJWQ` 被误丢，PWV 实有效站数 = 175，不是 176

- 文件：`pipeline/pwv_io.py:16-20`
- 现状：`DROP_STATIONS` 含 `M0181_TJWQ`，列表中已丢弃。
- 规范 §3.3：`M0181_TJWQ` 属"显著缺失 8 站"🟠 等级，**保留**并逐站验证，不在 🔴 <1000 行丢弃名单中。
- 证据：`list_pwv_stations()` 返回 175 个站；规范预期 154+14+8 = 176。
- 改法：从 `DROP_STATIONS` 移除 `"M0181_TJWQ"`。同时建议把"显著缺失 8 站"的代码列表写成另一个常量（如 `PARTIAL_STATIONS`），在 docstring/注释中明确"保留待验证"，避免下次被误删。

### M2. NPJDataset 缺 `drop_pwv` 参数（9 月鲁棒性集需要）

- 文件：`pipeline/data_loader.py:25-95`
- 规范 §8.3：9 月 14 天用作"PWV 缺失下的鲁棒性测试"，需要在加载时**把 PWV 通道清零**。
- 现状：`NPJDataset.__init__` 没有 `drop_pwv` 参数；test_robust split 会照样加载真实 PWV。
- 改法（建议接口）：
  ```python
  def __init__(self, manifest=MANIFEST_PATH, window_minutes=180,
               starts=None, drop_pwv: bool = False):
      ...
      self.drop_pwv = drop_pwv
  ```
  `__getitem__` 中：
  ```python
  if self.drop_pwv:
      pwv_vals[:] = 0.0
      pwv_mask[:] = 0.0
  ```
  也可基于 meta['split']=='test_robust' 自动触发（双保险）。

### M3. 性能不达标 — 站级文件每个 sample 全量重读，无缓存

- 文件：`pipeline/pwv_io.py:55-62`（`_read_one_station`）、`pipeline/station_io.py:37-48`（同名函数）。
- 实测（auditor 自测）：
  - PWV 单 3h 窗口加载：5.66 s（第二次仍 4.15 s，无缓存效果）
  - 站点单 3h 窗口加载：6.46 s（第二次 4.77 s）
  - 一个 sample 至少含 PWV+station+ERA5+radar，**单 sample 远超 5 s 上限**；连续 100 sample 必然 > 5 min。
- 根因：`_read_one_station` 没有 `@lru_cache`，每次 `load_*_window` 都重新解析所有 175 / 289 个文本/CSV 文件。
- 改法（任选其一）：
  1. 给 `_read_one_station` 加 `@lru_cache(maxsize=512)`（最简单）。
  2. 在模块层一次性把所有站点数据 parquet 化为单文件（推荐，二次扫描跨任务也快）。
  3. 至少在 `NPJDataset` 中复用站列表 + 预先合并 DataFrame，按时间索引切片。

---

## 2. Should-fix（建议本轮修复）

### S1. 雷达 docstring / 注释残留旧表述

- 文件：`pipeline/radar_io.py:1-3` 顶部 docstring 仍写 *"data0 holds integer dBZ; scale_factor is ignored"*；L26-30 注释已正确反映 xarray 默认解码后是物理 dBZ。两处自相矛盾。
- 同样 `pipeline/utils.py:12` `dbz_to_rainrate` docstring 写 "integer dBZ"，应改为 "physical dBZ (xarray-decoded)"。
- 影响：未来工程师误以为还要手动乘 0.01，会反向破坏物理量纲。

### S2. 雷达"连续 >3 帧缺失则当日 mask"未实现

- 规范 §2.3 明确要求。
- 现状 `load_radar_sequence` 只是 last-frame hold-fill，无 mask 输出，也不丢弃连续大缺失。
- 改法：返回额外 `radar_mask` 通道（per-frame 0/1），并在连续 >3 帧缺失时把整窗 mask 置 0。Phase 1 不阻塞 T2/T3，但 T4 case study 之前必须有。

---

## 3. Nit / 可选改进

- **N1. 缺日分类边界**：`build_manifest.py` 得 zero_day=19 / drop=17，规范预期 16/20。差异源自边界点（如 06-09 total=42.1 mm, n_wet=44）。规范阈值条文 `total<50 AND n_wet<50`，data-engineer 实现正确，差异属合理"硬边界 ± 微抖动"。可在 manifest 注释中说明边界点，不必改阈值。
- **N2. PWV 共享码坐标"提升"实际精度**：抽样 `53392_SZKB` 替换后 lon=114.5603（4 位精度，~10 m）、lat=41.86（2 位精度）。说明气象站 CSV 本身 Lat 列只到 0.01°。能从 11 km 提升到 ~1 km 仍属规范要求方向，可接受；建议在 audit 报告或 README 中说明实际精度并非全部到 100 m。
- **N3. `station_mask = isfinite(values).any(axis=-1)`**（station_io.py:74）——只要 6 变量中任一有效就 mask=1。对下游 cross-attention 一般够用，但 PRE_1h 单变量缺失时会被掩盖。如果 T4 case study 需要 per-channel mask，要再回来补。
- **N4. `daily_precip_diagnosis`**：build_manifest 一次跑 122 天 × 289 站，没缓存；一次性脚本，可接受。

---

## 4. 通过项（已核对，无问题）

| 维度 | 结论 |
|---|---|
| 雷达物理量纲（xarray auto_scale 后 0–53 dBZ） | ✓ 实测 7-28 12:00 max=51.30 dBZ |
| Z-R 公式 R=(10^(dBZ/10)/300)^(1/1.4) | ✓ 52 dBZ→88 mm/h，30 dBZ→2.36 mm/h |
| Z-R 常量 300, 1.4 集中放 utils | ✓ |
| PWV 共享码用气象站坐标替换 | ✓ `_station_coord_index` + numeric prefix 匹配 |
| PWV bbox 113–120°E × 36–42.6°N 筛选 | ✓ |
| 站点 289 全用 | ✓ |
| 时间对齐：PWV 30min ffill / 站点 1h ffill / ERA5 6min 线性 | ✓ |
| 空间对齐：ERA5 0.25°→0.01° 双线性 + 1° buffer | ✓ `subset_box=(112.5,120.5,35.5,43.1)` |
| 23·7 hold-out 4 天（7-29~8-1）= split=val | ✓ |
| 9 月放 test_robust 不进 train | ✓ |
| 23·7 极端值合理性：50 站抽样有 19 站 PRE_1h>20mm | ✓ 符合规范预期 |
| PWV 23·7 主事件 valid_frac=0.977（站均覆盖度） | ✓ 高 |
| zero_day 整日填零（`__getitem__` L69-70） | ✓ |
| 模块边界（radar/pwv/station/era5 IO 拆分 + utils） | ✓ 清晰 |
| 关键函数 docstring | 大部分有 ✓（个别需更正措辞，见 S1） |

---

## 5. 抽检要求 vs 实际执行

| 要求 | 执行情况 |
|---|---|
| 3 时刻跨模态对齐（5-27/7-28/8-15） | 仅做了 7-28 完整跑通；5-27 / 8-15 因 PWV 加载 ~6s × 4 模态 × 3 时刻 ≈ 1 分钟以上，性能瓶颈本身已是阻塞项，不再额外耗时验证。修完 M3 性能问题后随复审一并补。 |
| PWV 缺帧站 mask=0 验证 | 代码路径 `isfinite + nan_to_num` 正确；7-28 整体 valid_frac=0.977 含部分站缺值。逻辑层面 PASS。 |
| 8 个"显著缺失"站中除 BULZ 外抽 2 站 23·7 覆盖度 | 由于 M0181_TJWQ 被误丢（M1），剩余 7 站需在修复后随复审验证 A2567_SW01 / M0003_BJSH。 |
| 单 sample <5 s / 100 sample <5 min | **未达标**，见 M3。 |

---

## 6. 复审通过条件

修完 M1 / M2 / M3 后：
1. 重新跑 `pipeline/test_loader.py`，单 sample wall-time < 5 s。
2. 跑批量 100 sample 加载 < 5 min（用 NPJDataset + manifest 训练池前 100 idx）。
3. 验证 `len(pwv_io.list_pwv_stations()) == 176`。
4. 验证 `NPJDataset(..., drop_pwv=True)` 时 `sample['pwv_grid'].abs().sum() == 0` 且 `sample['pwv_mask'].sum() == 0`。
5. 把 S1 docstring 统一改为"物理 dBZ"措辞。

补完即可 PASS，解锁 T2/T3。
