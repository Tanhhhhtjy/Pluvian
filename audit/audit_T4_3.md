# Audit T4.3 — 训练脚本 + 6 个 Ablation Config

> **Finding A RESOLVED in commit 70e8b3a** (spot-check 2026-05-25 by auditor)
>
> training-engineer 在 4b9032e 之后追加了 follow-up commit 接通真实 MFD pipeline:
> - `pipeline/era5_io.py`:新增 `load_mfd_window(start, n_frames)`,从 `derived/mfd/` 加载 3 层(925/850/700),复用既有 bilinear weights 缓存 + linear_time_interp,结果落在 radar 网格 + 6-min cadence
> - `pipeline/data_loader.py`:`NPJDataset.__init__(load_mfd=False)` 新参数;True 时 `__getitem__` 注入 `era5_mfd` 张量(L109-111)
> - `scripts/train.py:85` `load_mfd = bool(cfg["model"]["mfd_channel_enabled"])` 联动两个 dataset;`split_batch`(L149-172) **优先用 `batch["era5_mfd"]`,只在 key 缺失时 fallback to zeros**(L165-172,带 channel pad/trim 守卫)
>
> **Spot-check 实跑(2026-05-25)**:
> ```
> NPJDataset(load_mfd=True)['2023-07-29T00:00:00']["era5_mfd"]
>   shape  : (30, 3, 661, 701)   ✓ 匹配 (T=30, L=3 levels, radar grid 661×701)
>   dtype  : torch.float32
>   means  : [5.68e-8, 9.13e-9, 2.34e-8]   三层均值各不相同
>   stds   : [2.14e-7, 1.59e-7, 9.97e-8]   三层方差各不相同
>   any non-zero: True            ✓ 真物理信号,非零
> ```
> abl_4 / abl_5 / abl_6 现在收到真实 MFD,实验有效性恢复。**Finding A 关闭**。
>
> Finding B (OOM `_oom_retry` dead code) 仍保留为非阻塞建议;GPU_SETUP.md 已写手动应对。
>
> ---

**Auditor**: auditor
**Date**: 2026-05-25
**Commit**: 4b9032e (919 lines), follow-up 70e8b3a
**Files**:
- `scripts/train.py` (594)
- `scripts/eval.py` (182)
- `configs/_base.yaml` (72) + 6 ablation YAML
**Verdict**: **PASS** (含 2 项 Finding — 1 项 research-validity 必修, 1 项功能缺失)

## 1. 6 个 Config 差异 vs Ablation 设计表

实测加载并 deep-merge 后的开关矩阵:

| Config                       | pwv | concat | era5 | mfd | budget | 设计意图 |
|------------------------------|-----|--------|------|-----|--------|----------|
| _base                        | 1   | 0      | 1    | 0   | 0      | 默认(=abl_3) |
| ablation_1_radar_only        | **0** | 0    | **0**| 0   | 0      | 纯雷达基线 ✓ |
| ablation_2_pwv_concat        | 1   | **1**  | 1    | 0   | 0      | I-2 反例: PWV 空间均值化 ✓ |
| ablation_3_pwv_channel       | 1   | 0      | 1    | 0   | 0      | I-2: PWV 作独立 cross-attn 流 ✓ |
| ablation_4_mfd               | 1   | 0      | 1    | **1**| 0     | I-2 主提案: +MFD 通道 ✓ |
| ablation_5_budget_loss       | 1   | 0      | 1    | 1   | **1**  | I-1: +水汽收支软约束 ✓ |
| ablation_6_full_pluvian      | 1   | 0      | 1    | 1   | 1      | 全模型 + 预留 diffusion ✓ |

**评价**:开关变化呈严格单调递增(monotonic ablation ladder),控制变量正确。`_base.yaml` 作为基线让 6 个文件每个只覆盖 4-8 行,符合 DRY,**通过**。

abl_6 与 abl_5 在 Phase 4 阶段开关全同,差异仅在预留 `diffusion:` block(L18-21,inert)——这是 Phase 6 占位,可接受,但 Phase 4 实验设计上 abl_5 / abl_6 是冗余跑(两份训练得相同模型)。**建议**:Phase 6 启动前在 abl_6 上加 `diffusion.enabled: true` 或干脆把 abl_6 推迟到 Phase 6 才注册到训练队列。

## 2. 训练循环正确性

### 2.1 Mixed Precision
`torch.autocast(device_type="cuda", dtype=dtype)` (train.py:309, 409) — bf16 默认,cfg 可切 fp16/fp32。**通过**。
`_base.yaml: precision: bf16` 与 A800 + cu121 默认 bf16 友好(无 GradScaler 需要)——正确选择。

### 2.2 Gradient Accumulation
- `loss_scaled = loss_total / grad_accum` (L334)
- `if (it+1) % grad_accum == 0: ... opt.step(); opt.zero_grad()` (L337-349)
- `total_steps = max_epochs * len(train_loader) // grad_accum` (L556) — 用于 cosine 调度的 step 计数,与实际 opt.step 次数一致 ✓

### 2.3 Gradient Clipping — **A4.2 建议已落实** ✓
```python
if grad_clip and grad_clip > 0:                                   # L338
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # L339
```
`_base.yaml: optim.grad_clip: 1.0` 与 A4.2 报告建议 `max_norm=1.0` 一致。**确认落实**。

### 2.4 Cosine + Warmup
`cosine_with_warmup` (L221-227) 标准实现:linear warmup → 半余弦衰减到 0。warmup_steps 由 epoch 数换算成 opt.step 数(L291),与 sched_step 单位一致。**通过**。

### 2.5 Checkpoint Top-K
`CkptManager` (L234-263):
- 每 epoch 保存 `epochNNN.pt` + 总是覆写 `last.pt` (L249) ✓
- 按 monitor 排序后取 top_k,余下 unlink (L258-262) ✓
- `best.pt` 是覆写(L256)不是符号链接,占额外磁盘(top_k+1 份),但保证 `best.pt` 总存在 ✓
- 防御:drop 列表里若包含 best 不会删(`if p.exists() and p != best`)——但因 records 已先按 score 排序、keep=records[:top_k]、best=records[0][1] 必在 keep 中,逻辑无 corner case。**通过**。

**小瑕疵**:`best.pt` 复写而非 symlink → 磁盘占 3 份 ckpt(top_k=3) + 1 份 best + 1 份 last = 5 份。base 模型 30M params * 4B + opt = ~250MB/份,~1.2GB/ablation × 6 = ~7GB,A800 数据盘 200GB 足够,但建议 Phase 6 训 large(72M)时改 symlink。非阻塞。

### 2.6 评估指标(validate)
`validate` 计算 CSI@{1,5,10,30mm}、FSS@{3,11}px、weighted_mse、CRPS-marginal、budget loss。
- `_csi`(L186-193): `denom > 0` 时返回 hits/(hits+fa+miss),否则 NaN。✓
- `_fss_binary`(L196-209): F.avg_pool2d 上算 fraction skill,与 `data_losses.FSSProxy` 数学等价但用 hard threshold(因为 val 不需梯度)。✓
- `_crps_marginal`(L212-214): 单成员 deterministic surrogate = MAE。注释明确说明这只是占位,Phase 6 ensemble 时再换。✓
- aggregate 用 `[v for v in vs if v == v]` 过滤 NaN(L438)。✓

## 3. Resume 功能 — 通过

train.py:529-535:
```python
state = torch.load(args.resume, map_location=device)
model.load_state_dict(state["model"])
opt.load_state_dict(state["opt"])
start_epoch = state.get("epoch", 0) + 1
global_step = state.get("global_step", 0)
```
保存(L572-578)与加载对称:`model`/`opt`/`epoch`/`global_step`/`config`/`metrics`。

**满足核心需求**。

**未保存**:
- RNG 状态(torch random / numpy random)——重续后无法 bit-exact 复现 dataloader shuffle 顺序。**实际影响小**(每 epoch 独立 shuffle 且 num_workers≥1 各自带 seed)。非阻塞。
- LR scheduler 状态:N/A — cosine 调度从 `global_step` 直接重算,无独立 state,resume 后调度连续 ✓

## 4. OOM 自动降 batch — **Finding B(未完全实现)**

审核要点 #4 说"OOM 自动降 batch"。实测代码:
- `_oom_retry` 函数定义在 L270-278:仅做 `torch.cuda.empty_cache()` + 重试,**不降 batch**
- 该函数在整个 train_one_epoch / validate 中 **未被调用**(grep 全文确认)

**结论**:OOM 处理是空头支票。建议:
- **方案 A**(快速,推荐):去掉 `_oom_retry` 死代码,在 GPU_SETUP.md 文档里写明 OOM 时手动改 config 的 `batch_size` 或 `grad_accum`(实际上 GPU_SETUP.md §6 已经这样写了 "如果 OOM,把 config 里的 batch_size 减半再试")。改文档比改代码风险小。
- **方案 B**:把 `_oom_retry` 真正包到 forward 调用上,且 OOM 时切到 `batch_size//2` 自动 retry,需要更复杂的状态机。Phase 4 不必上。

**当前评级**:此项功能未达,但因 GPU_SETUP.md 已有手动 OOM 应对说明 + 实际 A800 40GB 在 base 模型 batch_size=2 下不会 OOM,**降级为非阻塞 REVISE 建议**。

## 5. eval.py 独立可用 — 通过

`eval.py` 加载 `state["config"]`(train.py 已确保 ckpt 内嵌 cfg L577),独立重建 model + dataset,跑指定 split:
- 输出 `<out_dir>/metric.json` ✓
- 输出 `sample_{i}_<time>.png` 三 horizon 对比图(t+6, t+30, t+108 min,L74)✓
- 支持 `--drop_pwv` 用于 9 月鲁棒性测试 ✓
- 复用 train.py 的 `_collate / split_batch / _csi / _fss_binary / _crps_marginal`(L34-37),避免重复实现 ✓
- matplotlib 不可用时 graceful skip(L65-71)✓
- `--split` 参数允许跑 test_robust 等任意 manifest label ✓
- 提供 `weighted_mse / mae / crps / csi@{1,5,10,30} / fss@{3,11}` 9 个指标 ✓

**通过**。eval.py 与 train.py 的指标计算入口共用,不存在 "train 报 CSI A 数,eval 报 CSI B 数" 风险。

## 6. **Finding A — Research-validity 必修**(出自 commit msg 自白)

train.py:146-154:
```python
if mfd_channels > 0:
    # MFD is precomputed under /Data/tanh/npj/derived/mfd/ but not yet
    # piped through NPJDataset. Zero-pad to keep the architecture wired
    # — values are inert until the dataset starts emitting `era5_mfd`.
    model_in["era5_mfd"] = torch.zeros(...)
```
**问题**:`mfd_channel_enabled=True` 时模型确实收到 era5_mfd 张量(防止 A4.1 Finding 1 的 in_channels mismatch 崩溃),但全为零。
- abl_4 (`mfd=True, budget=False`) 在 MFD 全零时,等价于 abl_3 加 3 个零通道——**MFD 的物理信号完全无效**
- abl_5/abl_6 (`mfd=True, budget=True`) 中 MFD 零化,但 budget loss 仍然从 era5_fut 的真实 u/v/q 读取(train.py:322-330,**未受影响**),所以 budget 实验仍 valid
- **唯一受影响的是 ablation_4**:其 vs abl_3 的指标差将 ≈ 0,无法回答 "MFD 通道是否有用" 这个核心研究问题

**为什么算 Finding 而非 FAIL**:
1. commit msg 已显式声明这是 known-unimplemented,不是隐藏 bug
2. 不影响 ablation_1/2/3/5/6 的科学有效性
3. 修复路径清晰:`derived/mfd/` 已有 5 月×3 level 的 .nc 文件(A4.2 audit 已确认 compute_mfd.py 跑通)
4. 修复点窄:在 `NPJDataset.__getitem__` 加 ~10 行 mfd_io.load_mfd_window,再在 train.py:152 用 `batch.get("era5_mfd", zeros)` 替换硬零化

**必修要求**:T4.5 集成测试前(或 abl_4 训练启动前)必须接通 MFD pipeline,否则 abl_4 结果不可发表。

## 7. 其它正面观察(加分项)

1. **Config inheritance + deep_merge**(L43-62):简洁可读,避免 6 份 YAML 80% 重复
2. **window_minutes=180 拆 T_in=12 + T_out=18**(30 帧 6-min cadence=3h)与 RESEARCH_PROPOSAL "1h 输入 + 1.8h 输出" 吻合,且 `_starts_for_splits` 每 3h 滑动一次起点,**非重叠样本**,避免 leakage
3. **era5_fut 切片**(L133)给 budget loss 用,**未泄漏 ground truth radar 给 model_in**——data leakage 检查通过
4. **Save best symlink 替代 + drop 排除 best**(L260-262)逻辑虽冗余但安全
5. **`writer` 可选**(L541-546):tensorboard 缺失时 graceful skip,不阻塞训练
6. **AdamW betas=(0.9, 0.95)**(L519)——transformer 训练社区常见选择(GPT/LLaMA),比 (0.9, 0.999) 在大模型 + cosine LR 下更稳
7. **debug mode**:每 epoch 仅 5 batch + 1 epoch 退出(L376, L555, L585),冒烟测试快速
8. **CkptManager 防御**:`save_dir.mkdir(parents=True, exist_ok=True)` 容错重启
9. **`drop_pwv_val` 配置开关**(_base.yaml L29):为 9 月鲁棒性集 split 留接口,与 data_loader.py:83 `meta["split"]=="test_robust"` 路径协同

## 8. 小建议(非阻塞)

### S-1: budget_cfg["level_idx"]=1 对应 850 hPa 应文档化
train.py:322 `lvl = budget_cfg["level_idx"]` 取 era5_fut['u'][:,:,1]。lvl=1 假设 LEVELS=(925,850,700) 第二个是 850 hPa,与 compute_mfd.py:LEVELS 一致。但 `n_era5_levels=8`(_base.yaml L18),说明实际 ERA5 装载 8 层不是 3 层——`level_idx=1` 在 8 层语境下未必是 850 hPa。**建议**:_base.yaml 加注释 `# level_idx assumes era5_io ordering: 925/850/700/.../50, 850hPa = idx 1`,或在 train.py 验证 idx ↔ pressure 的映射。A4.5 需复核 era5_io 是不是按 sorted(LEVELS) 返回。

### S-2: `_oom_retry` 死代码清理
见 Finding B 方案 A,删除 L270-278 减少 grep 噪声,文档已在 GPU_SETUP.md 写明手动应对。

### S-3: total_steps 计算口径
L556 `total_steps = max_epochs * len(train_loader) // grad_accum` — 整除可能丢 0~grad_accum-1 个尾步,cosine 调度末尾会略微早归零。影响极小(<1% LR),非阻塞。

### S-4: eval.py 不计算 budget loss
eval.py 只跑 data 损失/指标,不复算 budget loss。若 paper Table 要展示 abl_5 / abl_6 的 budget residual 收敛情况,建议补 ~10 行。非阻塞,Phase 5 paper writing 时考虑。

### S-5: TensorBoard `writer.flush()` 缺失
train.py 在每个 step add_scalar 后没 flush,SummaryWriter 默认 ~120s 刷一次。长跑无碍,debug 短跑可能在 5 batch 退出时看不到曲线。非阻塞。

## 结论

**PASS**,前提是 **Finding A 在 T4.5 集成测试 / abl_4 真实训练启动前修复**。具体地:

| 维度 | 状态 |
|------|------|
| 6 config 差异 vs ablation 表 | ✓ 单调递增 ladder |
| Mixed precision | ✓ bf16 |
| Grad accumulation | ✓ |
| Grad clip = 1.0 (A4.2 建议) | ✓ **已落实** |
| Top-k checkpoint | ✓ 含 best/last |
| Resume | ✓ (RNG 状态未存,影响小) |
| OOM 自动降 batch | ⚠ Finding B:`_oom_retry` 未生效,文档替代,非阻塞 |
| eval.py 独立可用 | ✓ 含 metric.json + 3-horizon PNG |
| **MFD 通道真实接入** | ⚠ Finding A:**必修**,否则 abl_4 失效 |

T4.5 集成测试时:**优先**接通 NPJDataset → era5_mfd 输出,然后跑 abl_4 / abl_5 验证差异化训练曲线。否则 abl_4 vs abl_3 实验结果不可发表。
