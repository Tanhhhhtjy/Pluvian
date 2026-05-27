# Pluvian Handover — npj AI4NWP

> 切换 AI 时的接管文档。读完这一份 + `RESEARCH_PROPOSAL_v1.md` 即可继续工作。
> 最后更新：2026-05-27（ab3 训练至 ep45/60）

---

## 1. 一句话总览

在已有的「雷达 + 北斗 PWV」工作基础上，构建 **Pluvian**：一个多源耦合 + 物理约束的华北强对流降水 nowcasting 框架，6 组消融逐步加变量/损失，目标 **npj Climate and Atmospheric Science**。

---

## 2. 目标 & 约束

- **目标期刊**：Nature 子刊 npj（首选 *npj Climate and Atmospheric Science*）
- **创新点**：不能堆特征。每加一个变量必须有物理动机 + 消融证明。Cross-attention、water-vapor budget loss、GNSS-MFD 重建是潜在 novelty。
- **区域**：华北 113–120°E, 36–42.6°N（雷达边界），河北重点。
- **旗舰案例**：**23·7 京津冀暴雨**（2023-07-28 ~ 08-01），所有可视化围绕这个事件做。
- **不要做的事**：
  - 不要把工作做成 nowcasting baseline benchmark 刷分
  - 不要无脑加变量
  - 论文每个章节都先想"这里该有什么图"再写文字（用户原话）

---

## 3. 数据清单

`/Data/tanh/npj/` 下：

| 目录 | 大小 | 内容 | 时间 | 空间 |
|---|---|---|---|---|
| `radar_nc/` | 2.7 G | 24,144 NetCDF | 2023-05~09，6 min/帧 | 661×701, 0.01° |
| `pwv/` | 192 M | 216 北斗 PWV | 30 min | 176 站 |
| `stations/` | 120 M | 389 地面站 csv | 1 h | 289 站 |
| `era5/` | ~30 G | 4 vars × 8 levels | 1 h | 49×49, 0.25° |
| `derived/mfd/` | — | 预算的 Moisture Flux Divergence | 1 h | 925/850/700 hPa |

**注意**：
- 雷达只有有雨日，**无雨日没拷贝**，训练时需补零（见 `pipeline/radar_io.py`）
- 多尺度对齐（雷达 6 min / PWV 30 min / 站点 1 h / ERA5 1 h）本身就可能是方法学贡献点
- 详见 `pipeline/manifest.csv`（pre-built sample 索引）

---

## 4. 代码结构

```
model/
  pluvian.py        # 顶层模型 (radar encoder + fusion + decoder)
  encoder.py        # RadarEncoder (Swin-style) + PwvEncoder + Era5Encoder
  fusion.py         # PWV cross-attention 模块
  decoder.py        # PixelShuffle 上采样头
  losses/           # 加权 MSE + FSS + (可选) water-vapor budget

pipeline/
  data_loader.py    # NPJDataset：多源采样 + 时间窗口（已支持 load_era5=False 跳过加载）
  radar_io.py       # 6-min 雷达加载（缺日补零）
  pwv_io.py         # PWV 站点 → 稀疏格点
  station_io.py     # 地面站
  era5_io.py        # ERA5/MFD 加载（**关键**：保持原生 0.25°，不要 upsample 到雷达分辨率）
  utils.py          # RADAR_LAT/LON、时间插值

scripts/
  train.py          # 训练入口
  eval.py           # 离线评估
  watchdog.sh       # AutoDL 远程看门狗（已支持 config 参数）
  plot_prediction.py        # 单样本预测可视化（23·7 case 用）
  plot_ablation_compare.py  # ab1/ab2/... 学习曲线对比

configs/
  _base.yaml                       # 共享：batch_size=2, grad_accum=2, num_workers=2
  ablation_1_radar_only.yaml       # ✅ ab1 完成
  ablation_2_pwv_concat.yaml       # ✅ ab2 完成
  ablation_3_pwv_channel.yaml      # 🟡 ab3 训练中（cross-attn）
  ablation_4_mfd.yaml              # ⏳
  ablation_5_budget_loss.yaml      # ⏳ water-vapor budget physics loss
  ablation_6_full_pluvian.yaml     # ⏳ 全开
```

---

## 5. 训练 infrastructure（AutoDL 远程 A800 80 GB）

### SSH

```
host: connect.nma1.seetacloud.com
port: 54465
user: root
pass: 8agg9hbo3rhF
```

远程工程根目录：`/root/autodl-tmp/pluvian/`，python：`/root/miniconda3/bin/python`

### Watchdog（已部署）

`scripts/watchdog.sh`，作用：
1. 60 s 一查 `train.py` 是否活
2. 崩溃自动 resume from `last.pt`
3. 检测到 `[done]` → 调用 `shutdown -h now / kill -SIGTERM 1 / halt -f` **自动关 AutoDL 实例省费用**

**启动方式**（在远程）：
```bash
cd /root/autodl-tmp/pluvian && \
setsid bash scripts/watchdog.sh configs/ablation_X_xxx.yaml \
  < /dev/null > /dev/null 2>&1 & disown
```

`setsid` + 重定向是必须的，否则 SSH 断开会带走进程。

### Cron 轮询（Claude Code 内）

每 11 分钟（:07/:18/:29/:40/:51）SSH 抓取状态。模板：

```
grep -E "^\[ep|\[val\]" logs/ablation_X_xxx.log | tail -3
tail -3 logs/watchdog.log
ls ckpt/ablation_X_xxx/epoch*.pt | wc -l
nvidia-smi --query-gpu=power.draw --format=csv,noheader
```

完成时 watchdog 关机，cron 同时 `CronDelete`。

---

## 6. 实验进度

### ✅ ab1: radar only（baseline）

60 epoch 全跑完。final ep59：
- CSI@10mm = **0.596**
- CSI@30mm = **0.236**
- CRPS = 3.58

### ✅ ab2: + PWV channel-concat + ERA5

60 epoch 全跑完。final ep59：
- CSI@10mm = **0.619** (ab1: 0.596, +3.9%)
- CSI@30mm = **0.260** (ab1: 0.236, +10%)
- CRPS = 3.22

**结论**：naive 拼接给出小幅但一致的提升。多源信息没被充分利用。

### 🟡 ab3: + PWV cross-attention + ERA5（训练中，ep45/60）

最新 val (ep45)：
- CSI@10mm = 0.620 (≈ ab2)
- CSI@30mm = 0.260 (≈ ab2 final)
- CRPS = 3.26

**当前判读**：ab3 vs ab2 在 c10/c30 上 **基本持平**，CRPS 略劣。Cross-attention 在极端降水上没有数量级突破。可能 32 样本的 val set 噪声太大掩盖了改进，或者 cross-attn 让模型对 PWV 信号过度平滑。

剩 ~15 epoch (LR 衰减区) → 完成 ~16:00（2026-05-27 当天）。

### ⏳ ab4 / ab5 / ab6（未开始）

- **ab4**：+ MFD 通道（Moisture Flux Divergence at 925/850/700 hPa）。物理上比 ERA5 raw 更直接关联强对流，预期对 c30 有显著提升。
- **ab5**：+ water-vapor budget loss `∂PWV/∂t + ∇·(qV) ≈ −P`（differentiable 物理约束）。这是真正的 **novelty 点**——把物理方程做成损失项。
- **ab6**：全开 Pluvian（cross-attn + MFD + budget loss）。最终对照。

---

## 7. 关键性能 fix（不要回退！）

1. **ERA5 保持原生 0.25°**（`pipeline/era5_io.py`，2026-05-26）  
   之前 49×49 bilinear 上采样到 661×701 → 又被 Era5Encoder 下采样到 32×32，纯浪费。  
   修复后 epoch 时间从 60 min → 7 min（**~8× 加速**）。

2. **未启用 ERA5 时跳过加载**（`pipeline/data_loader.py`，`load_era5` 参数）  
   ab1 由此从 ~40 min → 7 min/epoch。

3. **DataLoader workers=2**  
   workers=8 会 RAM OOM（每个 worker 5 GB ERA5 cube）。

4. **batch_size=2 + grad_accum=2**  
   bs=4 在 A800 80 GB 会 VRAM OOM（特别是 ab2+）。effective batch = 4。

5. **不要 activation checkpoint**  
   之前试过，重算开销 > 内存收益。已 revert（commit 7303c7c）。

6. **train.py 输出 crop**  
   模型输出 664×704（8x pad），需 crop 到 661×701 才能算 loss。

---

## 8. 用户偏好（重要！）

来自 `~/.claude/projects/-Data-tanh-npj/memory/`：

### `feedback_pretraining_speed_check.md`
**启动每个新训练前必须短跑测速**：
- ~50-100 iter，与 baseline 对比 epoch time
- 慢 > 2× 必须停下沟通，不要 silent 烧钱
- 典型 silent bug：DataLoader 上采样、worker OOM、bf16 fallback fp32

### `feedback_visualization_priority.md`
**所有论文/汇报必须配套出版级可视化**：
- ≥300 dpi、行业标准 colormap（雷达用 NMC/CMA dBZ、MFD 用 RdBu_r、CAPE 用 YlOrRd、降水用 NWS）
- 一眼看懂结论
- 自动绘图脚本作为工具链一部分，不是事后手工补
- 风格基线：`figures/case_23p7/`（Phase 3 的 6 张诊断图）

### 沟通风格
- 用户喜欢简短回应、关键数字直给
- 关键判断必须坦率（"ab3 没显著突破"这种话直接说）
- 启训前必须先沟通（speed check）

---

## 9. 已知 gotchas / 坑

1. **`scripts/watchdog.sh` 旧版本硬编码 ab1 config**（已修复，新版接受 CLI 参数）。旧 commit 上的版本会让 watchdog 一启动就看 ab1 的 `[done]` 立刻关机。
2. **train.py 用 `--config` 不是 `-c`**。
3. **AutoDL SSH 偶尔 "Connection refused"**：通常是实例被自动关机（不活动超时）or watchdog 触发 shutdown。先 ping 端口，确认实例状态。
4. **每次 fresh 启动 ab*X* 之前清理 `ckpt/ablation_X_*/` 和 `logs/ablation_X_*.log`**，否则 watchdog 看到旧 `[done]` 立刻关机。
5. **`gh auth switch -u Tanhhhhtjy`** —— gh CLI 偶尔被其它会话切到别的账号，push 403 时检查。
6. **Loss 'budget=nan'** 是正常的（ab1-3 没启 budget loss）。
7. **CSI@30mm val 在前 20 epoch 抖动剧烈**（32 样本 val set 太小）。判断窗口是 ep30+。

---

## 10. 待办（按优先级）

> **2026-05-27 update**: 跑完代码内审 + 2024-2025 SOTA 扫描，发现现有架构 5 个 bug/缺陷。**ab4/5/6 暂停**，进入 Phase 7（详见第 14 节）。

1. **ab3 收尾**（当天完成）：
   - 拉回 `metric_log.json` + `best.pt` + log
   - 跑 `scripts/plot_ablation_compare.py`（扩展为 3 路曲线）
   - 跑 `scripts/plot_prediction.py` 生成 23·7 ab3 预测图
   - 推 GitHub
2. **进入 Phase 7a**（见第 14 节），**不要直接跑 ab4/5/6**
3. **Phase 6 评估准备**（在 Phase 7a 后做）：
   - CSI/FSS vs lead-time 曲线
   - 23·7 case study 时序栅格图
   - SOTA 对比表格（必须含 FusionCast / CasCast / DiffCast / NowcastNet）
4. **论文起草**：
   - Figure 1（事件解剖图，参考 `figures/case_23p7/`）
   - Method 章节里 cross-attn + budget loss 的数学公式 + 物理诠释
   - 必须先想图再写文字

---

## 11. 仓库 / GitHub

- 本地：`/Data/tanh/npj/`
- 远程：https://github.com/Tanhhhhtjy/Pluvian
- 当前分支：`phase5/training`
- 主分支：`master`
- 推送：`git push`（默认 phase5/training → origin）

`figures/` 目录里的 png 都已经 push 到 GitHub，可直接 URL 查看。

---

## 12. 联系上下文（Claude 记忆）

`/home/lih/.claude/projects/-Data-tanh-npj/memory/`：

```
MEMORY.md                              # 索引
project_npj_ai4nwp.md                  # 项目背景
project_data_inventory.md              # 数据清单
feedback_visualization_priority.md     # 可视化要求
feedback_pretraining_speed_check.md    # 启训前测速规则
```

切换 AI 后第一件事：**让新 AI 读这 5 个文件 + 本 HANDOVER.md**。

---

## 13. 紧急联系手册

| 状况 | 做法 |
|---|---|
| 训练崩溃 | watchdog 自动 resume；不用手动干预 |
| GPU 莫名关机 | watchdog 在监 `[done]`，是预期行为，省钱 |
| epoch time 突然变慢 > 2× | 停训，diagnose（多半 DataLoader 上采样回退或 ERA5 IO 缓存失效） |
| Val 指标抖动剧烈 | 32 样本 val set 噪声大，ep30+ 才能下定论 |
| SSH refused | 实例已关机，让用户重开 |
| OOM | 检查 batch_size / num_workers，看是不是 ERA5 cube 占爆 RAM |
| git push 403 | `gh auth switch -u Tanhhhhtjy` |

---

*文档维护：本文件随每个 ablation 完成后更新一次。如果你接管后跑了新实验，更新第 6 节并 commit。*

---

## 14. Phase 7 路线图（2026-05-27 audit 后新增）

跑完 ab1/2/3 后，并行做了内审 + SOTA 扫描，发现现有架构 5 个 bug/缺陷决定先暂停 ab4/5/6 转入 Phase 7。

### 14.1 必修 5 个 bug

| # | 缺陷 | 文件 | 后果 | 修复难度 |
|---|---|---|---|---|
| 1 | Val 集只有 32 样本（4 天 × 8 starts），CI ≈ ±0.05 | `pipeline/manifest.csv` | ab2 vs ab3 +0.02 在噪声里，所有排名不可信 | 1 天 |
| 2 | `_crps_marginal` 实际是 MAE 重命名 | `scripts/train.py:239` | 所有「CRPS」对比都是噪声 | 1 天（4-member MC-dropout） |
| 3 | PWV+station 同袋（176+289=465 tokens），无 PWV 自注意/时间混合 | `model/fusion.py`, `model/encoder.py:282-291` | cross-attn 退化成 average pooling，**ab3 ≈ ab2 的结构性原因** | 1 周 |
| 4 | Decoder = Linear(12→18) + 3× PixelShuffle，**无 U-Net skip** | `model/decoder.py:55,67` | 高分辨率结构在 head 前丢光，CSI@30mm plateau 真因 | 1 周 |
| 5 | Dead code 没接通：`FocalRainLoss`, 多阈值 FSS, `_oom_retry` | `model/losses/data_losses.py:138`, `train.py:189-194` | 已写好的损失项没生效 | 1 天 |

### 14.2 3 阶段路线

**Phase 7a（1 周）— 修 bug + 量化基线**
- 扩 val 集到 96+ 样本，CSI 改全集累计算（非 batch 平均）
- 修真 CRPS（4-member MC-dropout）
- 接通 dead code: `FocalRainLoss` (α=0.75, γ=2.0, thr=10mm) + 多阈值 FSS (10mm/30mm)
- 加 **Tweedie deviance loss** 替换部分 MSE (arXiv 2509.08369, +16% extreme rain MAE)
- 用现有 ab1/2/3 best.pt 在新 val 上重 evaluate，得带 CI 的真实排名

**Phase 7b（2-3 周）— 架构升级 v1（论文版本）**
- **Asymmetric gated radar-PWV fusion** 替换 cross-attn（对标 FusionCast arXiv 2603.13298）
- **MTLDM 风格 intensity-stratified decoder**（&lt;1, 1-8, 8-30, ≥30 mm/h 多头）
- **U-Net skip** 到 `radar_encoder.spatial1/spatial2`
- PWV 加 2 层 self-attn + temporal attention
- 重新设计 ablation matrix（旧 ab4/5/6 大概要重写）

**Phase 7c（3-4 周）— Diffusion 残差头 v2（reviewer revision 武器）**
- CasCast / DiffCast 风格 cascaded latent diffusion on residual（保留 7b backbone frozen）
- **PIANO 风格 ∂q/∂t + ∇·(qv) PDE loss**（arXiv 2512.01062）
- GenCast 风格 CRPS-direct training（可选）

### 14.3 必须 cite 的同期工作

- **FusionCast** (arXiv 2603.13298) — 最直接竞争对手，必须 head-to-head
- CasCast (arXiv 2402.04290, ICML'24)
- DiffCast (arXiv 2312.06734, CVPR'24)
- NowcastNet (Nature 2023)
- Aurora (arXiv 2405.13063, Nature 2024)
- GenCast (arXiv 2312.15796, Nature 2024)
- PIANO (arXiv 2512.01062)
- "Stop using RMSE" Hunt 2025 (arXiv 2509.08369)

### 14.4 Pluvian 的 novelty 边界（区别于 FusionCast）

候选差异点（论文要强调的）：
- Water-vapor budget loss（物理 PDE 约束）
- GNSS-MFD 重建（observation-driven physics constraint，潜在大 novelty）
- 多源融合包含 station + ERA5（FusionCast 只有 radar+PWV+future-prior）
- 不依赖 future-prior radar（FusionCast 关键依赖）
- npj Climate 期刊定位（vs FusionCast 是 arxiv 预印本）

详见 memory 文件：
- `~/.claude/projects/-Data-tanh-npj/memory/project_phase7_roadmap.md`
- `~/.claude/projects/-Data-tanh-npj/memory/reference_fusioncast_competitor.md`
