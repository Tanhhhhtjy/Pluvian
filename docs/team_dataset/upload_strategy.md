# 团队数据集 上传方案调研

> 调研人: Team B
> 日期: 2026-06-19
> 状态: lark-cli drive 上传方向有 **blocker**, 推荐方案见 §4

## 1. 数据规模 (基于真实 smoke test 50 帧)

| 项 | 单位 | 实测 |
|---|---|---|
| NPZ / 帧 | MB | 8.3 |
| PNG / 帧 (7 模态) | KB | ~46 × 7 = 322 |
| 单帧总 | MB | ~8.6 |
| 全集时刻数 (train+val+test_robust+event_test) | — | 13,536 |
| **全集总大小 (估)** | GB | **~116** |
| event_test 小集 (4 day × 144) | GB | ~4.9 |
| Wall-clock (8-worker, 单帧 ~1.15 帧/s) | h | ~3.3 |

PLAN.md 中 85 GB 偏低 (没算 sparse token + 实际 npz 压缩率), 以本表为准。

## 2. lark-cli 飞书云盘可行性 (已 verify)

### 2.1 命令矩阵

| 命令 | 用途 | 支持 ≥100MB 文件 |
|---|---|---|
| `lark-cli drive +upload --file ./xxx` | 单文件上传, >20MB 自动 multipart | 是 (chunked 3-step API) |
| `lark-cli drive +push --local-dir ./xx --folder-token ...` | 整目录 file-level mirror, 支持 `--if-exists=smart` 增量 | 是 (内部调 upload_all) |
| `lark-cli drive +sync` | 双向同步 | 是 |

### 2.2 路径限制 (踩到了)

`+upload --file` **只接受相对路径**, 必须 `cd` 到文件目录后用 `./xxx`. 写 wrapper 时要注意。

### 2.3 权限 (**blocker**, 实测确认)

`lark-cli auth status` 输出:
- bot identity: ready
- user identity: **missing (no user logged in)**

bot 试上传 6 B 测试文件结果:
```
authorization / app_scope_not_applied (99991672)
missing_scopes: ["drive:drive", "drive:file", "drive:file:upload"]
console_url: https://open.feishu.cn/app/cli_aaaedc41b9b8dce3/auth?q=drive%3Adrive%2Cdrive%3Afile%2Cdrive%3Afile%3Aupload
```

→ **lark-cli 上传链路当前不可用**, 需 user 二选一:
- A: 让管理员去开发者后台给 app `cli_aaaedc41b9b8dce3` 加 3 个 drive scope
- B: 跑 `lark-cli auth login` 用个人身份登录, 用 user identity 上传 (个人 drive 配额够不够要确认, 116 GB 不小)

## 3. 备选方案 (本机已具备的工具)

| 方案 | 优点 | 缺点 |
|---|---|---|
| `scp` / `rsync` 到同学机器 | 最简, 内网带宽全用上 | 需要同学开 ssh 入口 + 提供路径 |
| `rsync -P --partial` (断点续传) | 大文件友好 | 同上 |
| 打 tar.gz 分卷 (`tar czf - dir | split -b 5G - dataset.tgz.`) | 单包好 mirror, 可挂任何对象存储 | 解包不灵活, 增量不便 |
| 学校 / 公司 NAS / 内部对象存储 | 带宽高, 配额大 | 需 user 提供路径 |

本机 `/data4` 余量 1.5 TB, 116 GB 落地无压力。

## 4. 推荐方案 (MVP, 双线并行)

### 第 1 步: 小集先发, 让同学今天开工

```bash
# 4 个 storm day 的 event_test split, ~4.9 GB
python -m pluvian.team_dataset.run_batch \
  --out_root /data4/.../dataset_for_team_eventtest \
  --normalize_stats docs/team_dataset/normalize_stats.json \
  --splits event_test \
  --n_workers 8

# 打包成 zip (~4 GB 压完)
cd /data4/.../ && zip -r -1 dataset_for_team_eventtest.zip dataset_for_team_eventtest/
```

走以下 **任一** 通路 (按 user 确认顺序):
1. **lark-cli `auth login` 后用 user 身份 `drive +upload`** (最省事, 4 GB 走 multipart 没问题)
2. 或 scp 到同学机器
3. 或扔学校 NAS

### 第 2 步: 全集后台慢慢跑 + 同步

```bash
# 后台跑 (估 ~3.3 h)
nohup python -m pluvian.team_dataset.run_batch \
  --out_root /data4/.../dataset_for_team \
  --normalize_stats docs/team_dataset/normalize_stats.json \
  --splits event_test test_robust val train \
  --n_workers 8 > batch.log 2>&1 &

# 跑完后, 增量推到飞书 drive (前提 scope 已开)
cd /data4/.../ && lark-cli drive +push \
  --local-dir ./dataset_for_team \
  --folder-token <team_folder_token> \
  --if-exists=smart
```

`+push --if-exists=smart` 自带增量, 跑断重跑安全。

## 5. 待 user 决策的开放项

| # | 问题 |
|---|---|
| Q1 | 走飞书云盘 还是 scp / NAS? 如果飞书, 是去后台开 bot scope 还是用 user `auth login`? |
| Q2 | 如果飞书, 目标 folder_token 是什么? (root 还是某个团队目录) |
| Q3 | 同学机器是否开了 ssh? user / host / 路径? |
| Q4 | event_test 小集是周五晚发还是先等全集? |

## 6. 红绿黄

| 信号 | 状态 |
|---|---|
| `run_batch.py` 跑通 50 帧 | 绿 |
| 全集规模 (116 GB) 落地空间 | 绿 (1.5 T 余量) |
| lark-cli drive scope | **红** (bot 缺 3 个 scope, user 未登录) |
| 备选 scp / NAS 路径 | 黄 (等 user 提供) |
| MVP 小集 zip (event_test ~5 GB) | 绿 (本地可立即打) |
