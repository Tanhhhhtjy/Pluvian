# Pluvian 多模态短临降水数据集 — 使用指南

> 数据集版本: `team_dataset_v1`
> 时间范围: 2023-05-01 ~ 2023-09-30 (约 94 days, 13500 时刻)
> 空间范围: 京津冀+山东周边, 661×701 grid, 0.01° (~1km)
> 时间分辨率: 6 min (跟雷达对齐)
> 数据所有者: 田锦煜
> 用途: 小组同学训练 nowcasting 模型 + 期末汇报展示

---

## 1. 5 分钟上手

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载一帧 (单时刻包含 7 个 dense 模态 + sparse token)
d = np.load("npz/2023-07-30T09-00-00.npz")
print(list(d.keys()))

# 看雷达 (mm/h)
plt.imshow(d["radar_mmh"][::-1], cmap="turbo", vmin=0, vmax=20)
plt.title(f"radar mm/h @ {d['timestamp']}")
plt.colorbar(label="mm/h")
plt.show()
```

> ⚠️ npz 里数组是 **lat-major (661×701), lat 从南到北递增**。显示时上=北就 `[::-1]` 一下。

---

## 2. 目录结构

```
dataset_for_team/
├── README.md                        # 本文件
├── manifest.csv                     # 时间戳 + split 标签
├── normalize_stats.json             # 每模态 p2/p50/p98/min/max
├── png/                             # 8-bit 灰度展示图 (汇报展示用)
│   ├── radar/2023-07-30T09-00-00.png
│   ├── pwv_dense/...
│   ├── tem_station_dense/...
│   ├── rhu_station_dense/...
│   ├── prs_station_dense/...
│   ├── era5_t_925/...
│   └── era5_q_925/...
├── npz/                             # 训练用 float32 物理值
│   └── 2023-07-30T09-00-00.npz
├── stations_overlay/                # PNG + 真实站点散点 overlay
└── samples/                         # 几个 case 高清 PDF
```

每个 npz 是一帧, 包含全部 7 个 dense modality + sparse token。

---

## 3. 模态清单

| 模态 (npz key) | 物理单位 | 来源 | 原始分辨率 | dense 化方式 |
|---|---|---|---|---|
| `radar_mmh` | mm/h | 雷达组合反射率经 Z-R 公式 (Z=300 R^1.4) | 6 min, 661×701 | **原生 dense** |
| `pwv_dense` | kg/m² | GNSS 站 PWV (~70 站) | 30 min, sparse | IDW (k=8, power=2) |
| `tem_dense` | °C | 地面站气温 TEM (~389 站) | 1 h, sparse | IDW (k=8) |
| `rhu_dense` | % | 地面站相对湿度 RHU | 1 h, sparse | IDW (k=8) |
| `prs_dense` | hPa | 地面站气压 PRS | 1 h, sparse | IDW (k=8) |
| `era5_t_925` | K | ERA5 reanalysis t @ 925 hPa | 1 h, 0.25° | bilinear → 0.01° |
| `era5_q_925` | kg/kg | ERA5 reanalysis q @ 925 hPa | 1 h, 0.25° | bilinear → 0.01° |

### dense vs sparse 怎么选

| 你想要 | 用哪个 |
|---|---|
| 直接喂 CNN/UNet, 不想自己写 sparse-to-dense | 用 `*_dense` 字段 |
| 自己控制插值方法 (例如 GNN / Transformer over stations) | 用 `*_sparse_value` + `*_sparse_lonlat` + `*_sparse_mask` |
| 同时想要 dense backbone + sparse 真值 anchor | 两套都拿, IDW 插值结果配 sparse token 双路输入 |

> ⚠️ **诚信声明**: `pwv_dense / tem_dense / rhu_dense / prs_dense` 都是 sparse 站点 IDW **人工插值**得到的, **不是真实场**。在远离站点的 pixel, 数值仅是平滑外推, 不要当作 ground truth。`stations_overlay/` 里的 PNG 用红点叠了真实站点位置, 训练 reviewer 一眼能看出哪些是真观测。

---

## 4. normalize_stats.json 用法

文件长这样:

```json
{
  "radar":             {"p2": 0.0, "p98": 2.4, "log1p": true,  "p50": 0.1, "min": 0.0, "max": 6.8, "n_samples": 12345678},
  "pwv_dense":         {"p2": 8.2, "p98": 65.3, "log1p": false, ...},
  "tem_station_dense": {"p2": -3.1, "p98": 31.5, "log1p": false, ...},
  ...
}
```

每个模态:
- `p2 / p98`: dataset-wide 2nd / 98th percentile, 用作 PNG normalize 的 clip 边界
- `log1p`: 是否需要先 `log1p` 再 percentile (目前只有雷达=true, 因为 mm/h 长尾)
- `p50 / min / max / n_samples`: sanity check 用, 训练时一般用不到

### 推荐 normalize 写法 (PyTorch)

```python
import json
import numpy as np
import torch

with open("normalize_stats.json") as f:
    STATS = json.load(f)

def normalize(arr: np.ndarray, mode: str) -> torch.Tensor:
    s = STATS[mode]
    x = np.nan_to_num(arr, nan=0.0)
    if s["log1p"]:
        x = np.log1p(np.clip(x, 0, None))
    lo, hi = s["p2"], s["p98"]
    x = np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return torch.from_numpy(x.astype(np.float32))

d = np.load("npz/2023-07-30T09-00-00.npz")
radar_norm = normalize(d["radar_mmh"], "radar")  # in [0, 1]
```

PNG 文件就是按这套规则烤出来的 uint8, 想 sanity check 解码可以这样:

```python
from PIL import Image
png = np.array(Image.open("png/radar/2023-07-30T09-00-00.png"))  # uint8, 0~255
# PNG 上下翻转过 (北向上), 还原成 npz 的 lat-major:
png_latmajor = png[::-1]
```

---

## 5. manifest.csv 用法

```
date,split,zero_day,pre_total_mm,n_wet,starts_per_day
2023-05-01,train,False,,,24
2023-05-12,val,False,,,24
2023-05-17,test_robust,False,,,24
2023-05-06,drop,False,77.3,22.0,0
...
```

| split | 含义 |
|---|---|
| `train` | 训练集 |
| `val` | 验证集 (用于 model selection) |
| `test_robust` | 一般测试集 (混合天气) |
| `event_test` | 强对流个例测试集 (4 天 storm) — 汇报重点 |
| `drop` | **不要用**, 数据质量问题已弃 |

按 split 过滤示例:

```python
import pandas as pd
m = pd.read_csv("manifest.csv")
train_dates = m[m["split"] == "train"]["date"].tolist()
# 每天 144 帧 (6min cadence), 自己拼时间戳
train_ts = []
for d in train_dates:
    base = pd.Timestamp(d)
    for k in range(144):
        train_ts.append(base + pd.Timedelta(minutes=6 * k))
```

> 注意 `starts_per_day=8` 的日子是干旱日 (`zero_day=True`), 仍可用但抽样密度低。

---

## 6. 示例: 多模态拼图可视化

```python
import json
import numpy as np
import matplotlib.pyplot as plt

with open("normalize_stats.json") as f:
    STATS = json.load(f)

d = np.load("npz/2023-07-30T09-00-00.npz")

panels = [
    ("radar_mmh",       "radar (mm/h)",       "turbo"),
    ("pwv_dense",       "PWV (kg/m^2)",       "viridis"),
    ("tem_dense",       "T_station (deg C)",  "RdYlBu_r"),
    ("rhu_dense",       "RH_station (%)",     "BrBG"),
    ("prs_dense",       "P_station (hPa)",    "cividis"),
    ("era5_t_925",      "ERA5 t@925 (K)",     "RdYlBu_r"),
    ("era5_q_925",      "ERA5 q@925 (kg/kg)", "viridis"),
]

# 模态名 -> stats key (npz key 和 stats key 不同, 留意!)
NPZ_TO_STATS = {
    "radar_mmh":  "radar",
    "pwv_dense":  "pwv_dense",
    "tem_dense":  "tem_station_dense",
    "rhu_dense":  "rhu_station_dense",
    "prs_dense":  "prs_station_dense",
    "era5_t_925": "era5_t_925",
    "era5_q_925": "era5_q_925",
}

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.ravel()
for ax, (k, title, cm) in zip(axes, panels):
    s = STATS[NPZ_TO_STATS[k]]
    arr = d[k][::-1]  # 北向上
    if s["log1p"]:
        arr = np.log1p(np.clip(arr, 0, None))
    ax.imshow(arr, cmap=cm, vmin=s["p2"], vmax=s["p98"])
    ax.set_title(title)
    ax.axis("off")
axes[-1].axis("off")
plt.suptitle(f"multi-modal panel @ {d['timestamp']}")
plt.tight_layout()
plt.savefig("panel.png", dpi=120)
```

---

## 7. NPZ 完整 schema

每个 npz 文件包含以下字段:

```python
{
  # 元信息
  "timestamp":              str,                       # ISO 时间戳

  # dense 字段 (661, 701) float32, lat-major
  "radar_mmh":              np.ndarray,  # mm/h
  "pwv_dense":              np.ndarray,  # kg/m^2
  "tem_dense":              np.ndarray,  # deg C
  "rhu_dense":              np.ndarray,  # %
  "prs_dense":              np.ndarray,  # hPa
  "era5_t_925":             np.ndarray,  # K
  "era5_q_925":             np.ndarray,  # kg/kg

  # sparse PWV (N_pwv ~ 70)
  "pwv_sparse_value":       np.ndarray,  # (N,)   float32
  "pwv_sparse_mask":        np.ndarray,  # (N,)   float32, 1=该站有值
  "pwv_sparse_lonlat":      np.ndarray,  # (N, 2) float32, [lon, lat]

  # sparse 地面站 (N_sta ~ 389)
  "tem_sparse_value":       np.ndarray,  # (N,)
  "tem_sparse_mask":        np.ndarray,  # (N,)
  "rhu_sparse_value":       np.ndarray,
  "rhu_sparse_mask":        np.ndarray,
  "prs_sparse_value":       np.ndarray,
  "prs_sparse_mask":        np.ndarray,
  "station_sparse_lonlat":  np.ndarray,  # (N, 2) lon/lat, 跟上面 4 个 value/mask 共享
}
```

> 注意: PWV 站点和地面站是两套不同的网络, 坐标各自存 (`pwv_sparse_lonlat` vs `station_sparse_lonlat`)。

---

## 8. 常见坑

1. **npz dense 是 (661, 701) lat-major, lat 递增 (南→北)**
   不要 `transpose`, 显示时上=北就 `[::-1]` 一下。索引: `arr[i, j]` 对应 `lat = 36.00 + i * 0.01, lon = 113.00 + j * 0.01`。

2. **PNG 是 0-255 uint8, 不要拿来训练**
   PNG 是按 `normalize_stats.json` 烤出来的展示图, 已经 clip 到 p2/p98 + log1p (雷达)。要看物理值请用 npz, 要训练请用 npz + 自己 normalize。

3. **`station_mask` / `*_sparse_mask` 含义**
   `mask=1` 表示该站在该帧**有非缺失观测**, `mask=0` 表示该站此刻缺数据 (或被 QC 剔除)。dense 插值时只用 mask=1 的站点, 但 dense 输出本身**没有 mask** — 全场都被 IDW 填满了, 远离站点的值不可信。如果你需要"哪些 pixel 接近真站点", 自己用 `station_sparse_lonlat` 加距离阈值生成。

4. **雷达 mm/h 长尾分布**
   `radar_mmh` 多数时候 < 1 mm/h, 但暴雨核可以 > 50 mm/h, 直接做 percentile 会被 0 主导。所以 `normalize_stats.json` 里雷达的 `p2/p98` 都是在 **log1p 域**算的, 用的时候记得先 `np.log1p(np.clip(x, 0, None))` 再 normalize。

5. **缺帧 / 全零帧**
   雷达缺帧或 ERA5 缺文件的时刻**不会出现**在 npz/ 里 (build 阶段已经 skip)。所以 manifest 里 144 帧/天可能少几帧, 训练 dataloader 要按真实文件遍历, 不要硬算 144。

6. **时间戳文件名格式**
   文件名是 `YYYY-MM-DDTHH-MM-SS.npz` (用 `-` 不是 `:`, 因为 Windows 不支持冒号)。读 timestamp 用 npz 内的 `timestamp` 字段或者自己 `replace("-", ":", 2)` 后再 replace。

7. **多 worker dataloader**
   npz 是 `savez_compressed`, decode 有 CPU 开销, 建议 `num_workers >= 4`。

---

## 9. 数据集统计 (供 reviewer 参考)

- 总时刻数: ~13500 (94 days × 144 帧, 扣除缺帧)
- 总体 npz 大小: ~40 GB
- 总体 PNG 大小: ~43 GB
- split 分布: train ~74 days / val 8 / test_robust 8 / event_test 4 / drop 29 (不可用)
- 雷达 ZR 关系: `Z = 300 * R^1.4` (中国业务用)
- IDW 插值参数: k=8 近邻, power=2

---

## 10. 联系方式

数据问题 / schema 改动 / 字段缺失 → 找田锦煜。

引用本数据集请注明来自 `pluvian` 项目, 雷达数据来自中国气象局, ERA5 来自 ECMWF, GNSS PWV 处理自原始 troposphere zenith delay。
