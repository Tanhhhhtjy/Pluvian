# 多模态灰度数据集 — 团队交付方案

> Branch: `feature/team-dataset-multimodal-grayscale`
> Owner: 田锦煜
> Deadline: 周日汇报前
> Purpose: 给小组同学训练 + 汇报展示用的 5-modality 数据集

## 1. 任务定义

整理已处理好的**5 源数据**为灰度图 + NPZ，供小组同学训练 nowcasting 模型 + 汇报可视化展示。

5 个模态（user 决策："全包"）：

| 模态 | 来源 | 原始时间分辨率 | 原始空间形态 |
|---|---|---|---|
| 雷达 | `radar_nc/`，已过 dbz_to_rainrate | 6 min | dense grid 661×701, mm/h |
| PWV | `pwv/` GNSS 站点 | 30 min | sparse stations, kg/m² |
| 温度 TEM | `stations/station_*.csv` 第 12 列 + ERA5 t 925hPa | 1 hour / hour | sparse 289 stations + 0.25° ERA5 |
| 湿度 RHU | `stations/station_*.csv` 第 13 列（相对湿度 %） + ERA5 q 925hPa | 1 hour / hour | 同上 |
| 气压 PRS | `stations/station_*.csv` 第 9 列（hPa） + ERA5 surface_pressure (若有) | 1 hour / hour | 同上 |

注：地面站 ERA5 都要——地面站有真实点观测（physical truth），ERA5 提供 reanalysis full-grid 但是 0.25° coarse。两者互补。

## 2. 时间网格策略

所有模态对齐到**雷达 6 min 网格**（这是 nowcasting 主时间轴），所有窗口跟 phase7d manifest 一致（74 train / 8 val / 8 test_robust / 4 event_test = ~94 days × 144 frames/day = ~13500 时刻）。

不同模态的时间插值规则：

| 源 | 原 cadence | 插值到 6 min |
|---|---|---|
| 雷达 | 6 min | 直接用 |
| PWV | 30 min | 线性插值 (5 个 6-min 中间帧用前后 30 min 帧权重) |
| 地面站 | 1 hour | 线性插值 (10 个 6-min 中间帧) |
| ERA5 | 1 hour | 线性插值（pipeline 已有 linear_time_interp 复用） |

## 3. 空间网格策略

所有模态最终都生成 **661×701 dense grid** PNG 灰度图（汇报展示 + 同学直接 load 当 conv input）。

| 模态 | dense 化方式 |
|---|---|
| 雷达 | 原生 661×701 |
| PWV | 站点 → bilinear 插值到 661×701（参考 `pipeline/pwv_io.py` 已有 logic） |
| 地面站 TEM/RHU/PRS | 289 站点 → bilinear 插值到 661×701。空白区域 mask out |
| ERA5 t/q/surface_pressure | 0.25° → bilinear 上采样到 0.01° 661×701 |

⚠️ **重要诚信标注**：稀疏站点 → dense grid 是**人工插值**，**虚假信息**风险。
- PNG 图上叠加 **站点真实位置散点**（红点 / 黄点 overlay），让同学清楚哪些 pixel 是 truth，哪些是 interpolation
- NPZ 同时保留 **sparse token** 版本（value + lon + lat + valid_mask），让同学训练时自己决定要不要走 dense

## 4. 输出结构

```
/data4/WuMingrui/TianJinyu/npj/dataset_for_team/
├── README.md                        # 同学使用指南
├── manifest.csv                     # 时间戳 + split 标签
├── normalize_stats.json             # 5 模态的 dataset-wide p2/p98/min/max
├── png/                             # 展示 + 训练 (8-bit gray)
│   ├── radar/
│   │   └── 2023-07-29T00:00:00.png  # 661×701 grayscale
│   ├── pwv/
│   ├── temperature/  (地面站插值)
│   ├── temperature_era5_925/  (ERA5 t 925hPa)
│   ├── humidity/
│   ├── humidity_era5_925/
│   ├── pressure/  (地面站)
│   └── pressure_era5_surface/  (ERA5 surface_pressure if 有)
├── npz/                             # 训练用 float32 原值
│   ├── 2023-07-29T00:00:00.npz      # 一个 npz 含全 5 模态
│   └── ...
├── stations_overlay/                # PNG + 站点散点 overlay (汇报用)
│   └── ...
└── samples/                         # 几个代表性 case 高清 PDF (汇报贴文档)
    ├── case_2023-07-30T09_storm.pdf
    └── ...
```

NPZ schema (每帧一个文件):
```python
np.savez_compressed(
    "2023-07-29T00:00:00.npz",
    timestamp="2023-07-29T00:00:00",
    radar_mmh=...,          # (661, 701) float32 mm/h
    pwv_dense=...,          # (661, 701) float32 kg/m²，插值
    pwv_sparse_value=...,   # (N_pwv_station,) float32
    pwv_sparse_lonlat=...,  # (N_pwv_station, 2) float32
    tem_dense=...,          # (661, 701) float32 °C
    tem_sparse_value=...,
    tem_sparse_lonlat=...,
    rhu_dense=...,
    rhu_sparse_value=...,
    rhu_sparse_lonlat=...,
    prs_dense=...,
    prs_sparse_value=...,
    prs_sparse_lonlat=...,
    era5_t_925=...,         # (661, 701) float32 K
    era5_q_925=...,         # (661, 701) float32 kg/kg
    era5_sp=...,            # (661, 701) float32 Pa (if 有 surface_pressure 文件)
    station_mask=...,       # (661, 701) bool, 站点辐射范围
    split="train"|"val"|...,
)
```

## 5. 归一化策略

### 5.1 PNG (8-bit gray)
按"per-mode dataset-wide percentile clip + linear normalize"：
- 先 scan 全 dataset 算 p2 / p98 per modality
- 每帧 PNG = `clip(value, p2, p98) * 255 / (p98 - p2)` → uint8
- 雷达特殊：先 `log1p(value)` 再上面流程（压缩长尾）

### 5.2 NPZ (float32)
**不归一化**，保留物理原值（mm/h, kg/m², °C, %, hPa, K, kg/kg, Pa）。同学训练时根据需要 normalize。

### 5.3 normalize_stats.json
记录 dataset-wide p2/p50/p98/min/max + each modality 物理单位，供同学知道：
- log1p 该不该用
- 数值范围预期
- 是否需要 per-batch standardize

## 6. 数据量估算

| 项 | 数量 | 大小估算 |
|---|---|---|
| 时刻数 (94 days × 144 frames) | ~13500 | — |
| PNG / 模态 / 帧 (8 模态 × 13500) | 108,000 | ~400 KB × 108k = **43 GB** |
| NPZ / 帧 (5 dense + 3 ERA5 + sparse, 压缩) | 13500 | ~3 MB × 13500 = **40 GB** |
| Stations overlay (sample only, 100 case) | 100 | ~1 MB × 100 = 100 MB |
| Case PDF (汇报用 ~10) | 10 | ~50 MB |
| **总计** | | **~85 GB** |

⚠️ 这是远程上传规模。需要规划上传方式。

## 7. 工程拆解

| 阶段 | 任务 | 估时 |
|---|---|---|
| P1 | 写 `pluvian/team_dataset/build.py` 主流程：单帧 dump 函数（读 5 源 → 时间对齐 → 空间对齐 → 输出 PNG + NPZ） | 半天 |
| P2 | 先在 1 个 case 跑 + 肉眼 sanity check PNG | 1 小时 |
| P3 | 跑 normalize stats scan (全 dataset p2/p98) | ~1 h CPU |
| P4 | 全 dataset 批量 dump（8-worker multiprocessing，跟 pysteps wrapper 一样） | ~3-5 h wall-clock |
| P5 | 生成几个汇报 case PDF | 0.5 h |
| P6 | 写 README.md + normalize_stats.json + 同学使用说明 | 1 h |
| P7 | 上传到云盘 / 远程存储 | 取决于带宽 + 上传路径（需 user 提供） |

**关键时间表**：今天 + 明天写 P1-P3，周六全跑 P4，周日 P5/P6/P7。

## 8. 上传路径（等 user 确认）

需要 user 提供：
- 飞书云盘 / 公司内部 S3 / 学校 NAS / 公共网盘？
- 上传账号 / 路径 / 权限？
- 同学是否有 ssh 访问本机权限？如果有可以直接 rsync，不用上云

## 9. 风险登记

| 风险 | 应对 |
|---|---|
| 85GB 上传慢 / 带宽不够 | 分两轮：先把 event_test 4 storm 天 (~3 GB) 上传作为"小集"，让同学先开工；全集后台慢慢传 |
| 站点稀疏插值的虚假信息 | PNG 上 overlay 真实站点散点 + NPZ 保留 sparse token + README 显式标注 |
| ERA5 surface_pressure 文件是否存在 | 先 check `ls era5/era5_sp*.nc`，没有就跳过这个模态 |
| station CSV 时间对齐 (1h cadence → 6 min) | 用 pipeline/station_io.py 现有 linear interp |
| 同学 npz load 时 schema 变化 | 在 README 锁定 schema 版本号，dump 完不再改 |

## 10. 验收标准

- ✅ 90% 时刻有 5 模态非空数据
- ✅ PNG 肉眼可读，强对流核明显
- ✅ NPZ float32 物理值，单位文档清楚
- ✅ Stations overlay PNG 至少 10 个 case 可供汇报
- ✅ README 含"5 分钟上手"示例 load 代码
- ✅ Normalize stats JSON 含每模态 p2/p50/p98/min/max
- ✅ 上传链接发送给小组同学
