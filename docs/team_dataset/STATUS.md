# Team Dataset 集成状态报告

> Judge 集成时间: 2026-06-19
> Branch: `feature/team-dataset-multimodal-grayscale`
> 用途: 周日汇报数据集交付的总状态 + user 拍板项

## 1. 三 Team 模块状态

| Team | 交付物 | 状态 | 备注 |
|---|---|---|---|
| A: build.py | 单帧 7 模态 dump (PNG+NPZ) | 绿 | 真 stats 单帧 8.6 MB, smoke 视觉合理 |
| B: run_batch.py | 8-worker 调度 + run_log + 上传调研 | 绿 (代码) / **红 (上传)** | 飞书 drive scope 未开, user blocker |
| C: scan_stats.py + README | dataset-wide stats + 同学使用文档 | 绿 | 120 帧抽样 5.5 min 跑完, 7 模态全有数值 |

## 2. Normalize Stats (本次实测)

抽样: 120 帧 (sample_rate=0.2, hours/day=6, 4 worker), 失败 19 帧 (radar 缺帧/ERA5 边界), 净 101 帧 × 7 模态 × 10000 px ≈ 7 M sample, 5.5 min 跑完。

| 模态 | p2 | p50 | p98 | 单位 | log1p |
|---|---|---|---|---|---|
| radar | 0.0169 | 0.0169 | 0.963 | log1p(mm/h) | yes |
| pwv_dense | 9.07 | 33.97 | 67.50 | kg/m² | no |
| tem_station_dense | 12.29 | 24.20 | 34.33 | °C | no |
| rhu_station_dense | 19.13 | 69.83 | 97.72 | % | no |
| prs_station_dense | 853.0 | 980.6 | 1011 | hPa | no |
| era5_t_925 | 287.3 | 296.3 | 303.4 | K | no |
| era5_q_925 | 0.00276 | 0.01133 | 0.01878 | kg/kg | no |

物理 sanity:
- 雷达 p98 log1p=0.96 → ~1.6 mm/h, 强对流核会饱和到白 (符合预期, 弱降水细节保留)
- ERA5 t@925: 287-303 K = 14-30 °C, 夏季华南合理
- ERA5 q@925: 2.8-19 g/kg, 925 hPa 比湿合理
- 气压 853 hPa p2 异常低: 由 IDW dense 化时高地形外推 + 891 站夹杂的实际低压站, 训练时建议同学用 sparse token

## 3. Single-frame Build 验证 (用真 stats)

2023-07-30T09:00:00 (storm), 7 模态 PNG 全有可读对比:

| 模态 | PNG min/max/mean | 评价 |
|---|---|---|
| radar | 0/255/69 | 强核饱和, 大量背景 0, OK |
| pwv_dense | 63/255/210 | 水汽场全图都有量 |
| tem_station_dense | 22/226/146 | 灰度全段使用 |
| rhu_station_dense | 90/255/213 | 高湿区饱和 |
| prs_station_dense | 0/254/167 | 边界外 IDW 留 0, 训练可用 station_mask 过滤 |
| era5_t_925 | 100/255/143 | 温度场顺梯度 |
| era5_q_925 | 110/255/208 | 湿度场顺梯度 |

未出现全黑/全白, normalize stats 落地正确。

## 4. event_test MVP 跑通 (576 帧 4 storm days)

```
ok=572 / skip=4 / elapsed=340.8s / out=/data4/WuMingrui/TianJinyu/npj/dataset_for_team_mvp
```

| 指标 | 实测 |
|---|---|
| n_ok | 572 / 576 (99.3%) |
| 跑时 (8-worker) | 5.7 min wall-clock |
| 平均速率 | 1.69 帧/s |
| 总大小 | 4.9 GB (NPZ 4.7 G + PNG 245 M) |
| 失败原因 | 2023-07-31 07:48-08:06 (radar 4 帧缺失) — 上游数据问题, 非代码 bug |

随机 3 帧 (2023-07-29T12, 2023-07-30T09, 2023-08-01T06) 灰度分布健康, 强降水帧 radar mean=11-77 std=46-100, 符合 storm vs lull 差异。

## 5. 全集 ETA (待 user GO)

| 项 | 数值 |
|---|---|
| 全集时刻数 (event_test + test_robust + val + train) | 13,536 |
| 8-worker 实测吞吐 | 1.69 帧/s |
| **wall-clock ETA** | **13536 / 1.69 / 60 ≈ 133 min ≈ 2.2 h** (比 PLAN 估的 3.3 h 还快, 因为之前估算保守) |
| 输出大小 (8.6 MB/帧 × 13500) | ~116 GB |
| `/data4` 余量 | 1.5 TB, 够用 |

跑法 (已就绪, 等 user GO):
```bash
nohup /home/WuMingrui/miniconda3/envs/npj/bin/python -m pluvian.team_dataset.run_batch \
  --out_root /data4/WuMingrui/TianJinyu/npj/dataset_for_team \
  --normalize_stats docs/team_dataset/normalize_stats.json \
  --splits event_test test_robust val train \
  --n_workers 8 > /tmp/dataset_full_run.log 2>&1 &
```

## 6. User Blocker (必须 user 拍板)

### Blocker 1: 上传通路 (最关键)

`lark-cli drive +upload` 用 bot 身份缺 3 个 scope:
```
missing_scopes: ["drive:drive", "drive:file", "drive:file:upload"]
console_url: https://open.feishu.cn/app/cli_aaaedc41b9b8dce3/auth?q=drive%3Adrive%2Cdrive%3Afile%2Cdrive%3Afile%3Aupload
```

User 三选一:
- **A**: 去飞书后台给 app `cli_aaaedc41b9b8dce3` 加 3 个 drive scope (~5 min)
- **B**: `lark-cli auth login` 用个人身份登录, 用 user identity 上传 (个人 drive 配额 116 GB 是否够要确认)
- **C**: 给个 scp / NAS / rsync 路径, 不走飞书

### Blocker 2: 全集跑不跑 GO 信号

- 跑全集 ~2.2 h CPU, 占 116 GB 磁盘
- 现在等 user 一句 "上" 就启
- 或者只跑 MVP (event_test 5 GB 已就绪), 全集等周六

### User 拍板后 24 h 能完成的事

| 拍板项 | 拍板后我能做什么 |
|---|---|
| Blocker 1 选 A (开 scope) | scope 生效后立即 `drive +push` MVP + 全集 |
| Blocker 1 选 B (auth login) | login 完成后立即 push, 走个人 drive |
| Blocker 1 选 C (scp/NAS) | 给路径后立即 rsync, 不卡 |
| Blocker 2 GO | 启全集后台跑, ETA 2.2 h |

## 7. 周日汇报 Deliverable Checklist

| # | 项 | 状态 |
|---|---|---|
| 1 | normalize_stats.json (7 模态 dataset-wide) | 完成 |
| 2 | event_test MVP (576 帧, 5 GB) | 完成 |
| 3 | README.md 同学使用文档 | Team C 已交 |
| 4 | upload_strategy.md 调研 | Team B 已交 |
| 5 | 全集 13500 帧 dump (~116 GB) | 等 user GO |
| 6 | Stations overlay PNG (汇报展示, 10 case) | 待做 (半小时) |
| 7 | Case PDF (storm 高清, 汇报贴文档) | 待做 (半小时) |
| 8 | 上传到团队空间 + 给同学链接 | 等 Blocker 1 |

## 8. 已知遗留 / 后续建议

- `prs_station_dense` IDW 在边界外是 0, 影响 PNG 视觉一致性。建议训练同学用 `station_mask` 过滤 (NPZ 内已存)。
- 2023-07-31 08:00 前后 radar 4 帧缺失, 全集跑时会有类似零星 skip (历史经验 < 5%)。
- scan_stats 121 帧中 19 帧 hard skip (15.7%), 主要是 ERA5 边界 / radar 缺帧, 用 sample_rate=1.0 全跑会拿到更稳的 p2/p98, 但当前数值已够 PNG 归一化用, 不必重跑。

## 9. 等 user 的两个核心问题

1. **上传通路: 飞书 (开 scope / auth login) / scp / NAS?**
2. **全集 13500 帧 GO 不 GO?** (跑了占 116 GB 磁盘, 2.2 h CPU)
