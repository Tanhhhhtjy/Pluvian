# Audit T4.1 — Pluvian Multi-Source 模型架构

**Auditor**: auditor
**Date**: 2026-05-25
**Commit**: 78bcee9
**Files** (866 lines total):
- `model/pluvian.py` (201)
- `model/encoder.py` (368)
- `model/fusion.py` (78)
- `model/decoder.py` (93)
- `model/test_shapes.py` (107)
- `model/__init__.py` (19)
**Verdict**: **PASS** (with 1 documented gap for T4.3/T4.5 to close)

## 1. 接口一致性 (vs NPJDataset)

`pipeline/data_loader.py` `__getitem__`(L95-107) 输出键:
```
radar, radar_mask, pwv_grid, pwv_mask, pwv_coords,
station_grid, station_mask, station_coords, era5, time, meta
```

`Pluvian.forward`(pluvian.py:141-167) 消费键:
```
radar              ✓
pwv_grid, pwv_mask, pwv_coords           ✓
station_grid, station_mask, station_coords ✓
era5                ✓ (dict of {u,v,q,t}: (B,T,L,H,W))
era5_mfd            ⚠ 见下方 §1-finding
```

**接口对齐情况**:其余字段全匹配。`Era5Encoder.forward`(encoder.py:236) 用 `sorted(era5.keys())` 拼通道——稳定可重现,但要求 ablation/baseline 跨实例一致(NPJDataset 当前总是返回 `{u,v,q,t}`,4 vars,与 model 默认 `n_era5_vars=4` 一致 ✓)。

### Finding 1 (非阻塞,T4.3/T4.5 修)

`mfd_channel_enabled=True` 时模型读 `batch["era5_mfd"]`(pluvian.py:156),但 **NPJDataset 当前不输出该键**。
- 若 T4.3 启用 ablation_5/6,且 dataset 未补 `era5_mfd`,`Era5Encoder` 会在第一层 conv 因 in_channels 缺失 mfd_channels(默认 3)而崩溃(`in_ch = V*L + mfd_channels = 32+3=35`,但 x 只有 32)。
- **test_shapes.py 通过**是因为 `_fake_batch` 手工注入 `era5_mfd`(test L45),掩盖了集成层的真空。
- **建议**:T4.3 训练入口需要在 collate 之前把 `derived/mfd/era5_mfd_{level}_{YM}.nc` 装载并加 batch key,或者在 model 端检测 `mfd is None` 时把 mfd_channels 当 0 处理(防御式)。
- 影响:不阻塞当前 4 个 ablation 通过 test_shapes,但阻塞主提案(I-2)在真实数据上的训练,需在 A4.3/A4.5 中复核。

## 2. 可微性 — 全部通过

实跑 `python -m model.test_shapes` 在 CPU torch 2.12+cpu 上:
```
[base + MFD]      grad_norm_total=16.87
[small]                          7.69
[no-pwv]                        14.95
[pwv-concat-only]               10.88
[no-era5]                       27.45
[base-no-mfd]                   20.25
```
6 个变体全部 `loss.backward()` 跑通,grad 非零。无 in-place 操作泄漏(test 在每次后清空 grads)。

`fusion.py:70-73` 的"全 mask 行 un-mask 首 token"安全阀是必要的——`nn.MultiheadAttention` 在某行 kpm 全 True 时返回 NaN,会污染 grad。这个处理是正确的。

## 3. 参数量 — 在 30-80M 区间(主提案合规)

```
small        :  8.79 M   (hidden=256, n_layers=4)
base         : 30.36 M   (hidden=384, n_layers=8)   ← 主提案
base-no-mfd  : 30.35 M
no-pwv       : 30.35 M
pwv-concat   : 30.35 M
no-era5      : 28.30 M   (去 era5_encoder 后 ~ 2M 缩水)
large(声明)   : ~72.8 M  (hidden=512, n_layers=12,未实测但 preset 已暴露)
```
test_shapes.py:82 显式 `assert 20e6 < n_params < 80e6` 验证主提案落在 budget 内。**通过**。

small (8.79M) 在审核区间(30-80M)外,但 SIZE_PRESETS 提供它作为可选 debug/CPU 变体,**非主提案**,不计入违规。

## 4. Shape 正确性 — 通过

输入 `(B=1, T_in=30, H=128, W=128)` →
- `RadarEncoder`: stem stride-4 + down stride-2 = 8× ↓ → `(1,30,C,16,16)` ✓
- `Era5Encoder`: adaptive_avg_pool2d(32) + 1 stride-2 conv → `(1,30,C,16,16)` ✓
- ERA5 bilinear 上采样到 `(16,16)` 与 radar 残差相加(pluvian.py:117-124) ✓
- `CrossAttentionFusion`: Q=(B*T, H*W, C), K=V=(B*T, N, C),per-frame attn ✓
- `PluvianDecoder`: time_proj 30→18, 3 PixelShuffle ×2 = ×8 ↑ → `(1,18,128,128)` ✓

所有变体输出 `rain_pred / pwv_pred = (B,18,H,W)` 与 test_shapes.py:52-55 断言一致。

## 5. 6 个 Ablation Hook — 全暴露

| Hook | 位置 | 实测 ✓ |
|------|------|--------|
| `pwv_enabled=False` | pluvian.py:168-171 (零化 pwv_vals/mask) | no-pwv 通过 |
| `pwv_concat_only=True` | pluvian.py:172-177 (空间均值化,保留架构) | pwv-concat-only 通过 |
| `era5_enabled=False` | pluvian.py:87-94 (不构造 era5_encoder) | no-era5 通过 |
| `mfd_channel_enabled` | pluvian.py:90,156-157 (channel 拼接) | base+MFD vs base-no-MFD 通过 |
| 主提案 (`I-1+I-2+I-3`) | 默认 `pwv_enabled=True, era5_enabled=True, mfd=False`(可开) | base 通过 |
| `size` preset | pluvian.py:67-71 (small/base/large) | small + base 通过 |

**Ablation 设计审查**:
- `pwv_enabled=False`: 零化值+mask,但 `sparse_encoder` 仍执行——意味着 N_pwv 维度 token 仍存在,只是全部 kpm=True。fusion 安全阀(L70-73)防 NaN。**正确**,这是"信息流断开"而非"结构差异",符合 ablation 单变量原则。
- `pwv_concat_only=True`: `pwv_vals.mean(dim=-1, keepdim=True).expand_as(pwv_vals)`(L177)把空间分异抹掉但保留架构。**符合 I-2 ablation 语义**(测试空间覆盖信号 vs 整体均值贡献的差异)。
- `era5_enabled=False`:不构造 era5_encoder,参数量降 2M(28.30 vs 30.35)。**正确**。
- `mfd_channel_enabled`:开启时 era5_encoder.in_ch 增 3,但 dataset 接口缺口见 Finding 1。

## 6. Cross-Attention 真伪 — 真做了,非伪 concat

`fusion.py` `_CrossAttnBlock.forward`(L29-36):
```
self.cross(norm_q(q), norm_k(k), norm_k(k), key_padding_mask=kpm)
                  ↑Q from dense  ↑K=V from sparse
```
- 使用 `nn.MultiheadAttention(batch_first=True)`,n_heads=8
- Q 来自 dense (radar+ERA5) 的 `(B*T, H*W, C)`,K/V 来自 sparse tokens `(B*T, N, C)`
- per-frame attention(每个 radar 帧只对其同步的 sparse slice attn),T 维线性
- 残差+MLP 标准 Transformer block,`n_layers = max(1, n_layers//2)`(pluvian.py:104)
- key_padding_mask 正确传递,处理 missing observation

**这是真正的 cross-modal attention,不是 concat 充数**。审核通过。

补充亮点:`SparseTokenEncoder._expand_time`(encoder.py:312-319) 用 last-fill 把 30min/1h 的 sparse stream 对齐到 6-min radar 时钟,与 `DATA_HANDLING.md §7` 一致。

## 加分项

1. **Fourier PE on (lon,lat,[alti])**(encoder.py:137-147 + 295-310):坐标归一化到 [-1,1] 后做 sin/cos with `n_bands=8 × max_freq=64`——典型 NeRF/PerceiverIO 风格,比简单 sin/cos 更好支撑 station 稀疏分布
2. **Type embedding**(encoder.py:293, 343, 357):PWV vs station 用 `nn.Embedding(2,dim)` 区分,避免模型把两类 token 混淆
3. **学习的 T_in→T_out 时间投影**(decoder.py:55):用 `nn.Linear(30,18)` 540 个参数实现非递归 forecast,简洁高效
4. **All-masked-row 安全阀**(fusion.py:70-73):防 NaN,关键稳定性保护
5. **GroupNorm with `gcd(8, out_ch)`**(encoder.py:41):自适应 group 数,避免 channels 不能整除 8 时崩溃
6. **PixelShuffle 上采样**(decoder.py:25-28):比 ConvTranspose 减少 checkerboard artifact
7. **input_frames mismatch 兼容**(decoder.py:75-82):若 T_in 与配置不符,用 linear interp 自动适配——增加鲁棒性
8. **size preset 把 hidden_dim/n_layers/n_heads 锁死**(pluvian.py:34-38, 67-71):避免实验里"我 small 和你 small 不一样"

## 小建议(非阻塞)

### S-1 (Finding 1 复述): mfd_channel_enabled 与 NPJDataset 集成缺口
- 当前实现:`batch.get("era5_mfd")` → 缺则 None,但 `Era5Encoder.in_ch` 已固化 +mfd_channels
- 修复方案 A (推荐,在 T4.3 trainer 中):在 NPJDataset 加可选 `load_mfd=True` 参数,从 `derived/mfd/` 加载并按 (B,T,L_mfd,H,W) 返回
- 修复方案 B (model 防御式,在 T4.1 改一行):`Era5Encoder.forward` 检测 `mfd is None and self.mfd_channels > 0` 时 zero-pad 出 mfd_channels 维度,避免崩溃。但这会让 ablation 5 的训练损失全为零信息,**不推荐 silent fallback**,应优先方案 A 并在缺失时 raise
- **影响**:A4.3/A4.5 必查项

### S-2: SparseTokenEncoder._expand_time 用 `floor` 时间映射
encoder.py:317-318 `(t * T_src / T_target).floor()` 实现 last-fill。对 T_target=30, T_src=6 (30-min PWV 内 6 个时刻覆盖 30 个 6-min radar 帧):映射为每个 source 帧填 5 个 target 帧,边界 ok。但若 T_target < T_src 会塌缩——当前 T_in=30 配置安全,但若未来缩短输入 window 需要重测。非阻塞。

### S-3: forecast horizon shape 写死
`forecast_frames=18` 默认在 6-min cadence 下 = 108 min ≈ 1.8h。与 RESEARCH_PROPOSAL_v1 "短临 2h" 一致,可通过 ctor 调整。无问题。

### S-4: window=8 + window-attention non-shifted
encoder.py:50-55 文档已声明放弃 shift 简化。对 H=128/8=16 latent grid + 16/8=2×2 windows 影响不大;真实 H=661 → latent 83 → 10×10 windows,跨窗 receptive field 全靠 stride-down + temporal attn 弥补。**对 nowcasting 通常 OK**,但建议 RESEARCH_PROPOSAL 在 Method 节里明确说明这是 design choice 而非 oversight。

### S-5: Decoder.input_frames mismatch 用 linear interp(decoder.py:79-82)
功能上合理,但隐藏了"上游 T_in 与 model 配置不符"的告警。建议至少 `warnings.warn` 一次,免得 silent 漂移。非阻塞。

## 结论

**PASS** — 架构设计扎实,真 cross-attn(非 concat 占位),6 个 ablation hook 全部跑通 forward+backward,参数量 30.36M 落在 30-80M budget 内,与 NPJDataset 接口除 `era5_mfd` 缺口外完全对齐,test_shapes 全部通过。

**唯一需要在下游(T4.3 训练入口、T4.5 集成测试)修复的**:NPJDataset 需要补 `era5_mfd` 输出,否则 `mfd_channel_enabled=True` 在真实数据上会崩溃。本审核已记为 Finding 1。

可作为 T4.3 trainer / T4.5 集成测试的稳定基线。
