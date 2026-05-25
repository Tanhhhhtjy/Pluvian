# Audit T4.2 — 水汽收支损失模块

**Auditor**: auditor
**Date**: 2026-05-25
**Commit**: 0784a90
**Files**:
- `model/losses/water_budget.py` (323 lines)
- `model/losses/data_losses.py` (172 lines)
- `model/losses/test_water_budget.py` (319 lines)
- `model/losses/__init__.py`
**Verdict**: **PASS**

## 1. 物理正确性

### 1.1 球面 FD 因子 — 正确
`SphericalDivergence._ddx/_ddy`(water_budget.py:109-134) 与 `pipeline/compute_mfd.py:_spherical_divergence`(L40-63) 公式一一对应:
- `dlon = mean(diff(lon)) * π/180`,`dlat = mean(diff(lat)) * π/180`
- `dx = R·cos(lat)·dlon` 随纬度变化(broadcast 到 `(H,1)`)
- `dy = R·dlat` 常量,R=6_371_000 m
- 内部中心差,边界单侧差(forward at col 0, backward at col -1)

Lat 升序断言(L154-155)与上游 `era5_io._load_monthly` 约定一致。

### 1.2 单位转换 — 正确且文档清晰
水汽收支方程: `d(PWV)/dt + ∇·(qV)_col = -P + E`

代码路径(water_budget.py:278-287):
```
qu, qv  : kg/kg · m/s = m/s
div     : 1/s  (q is dimensionless)
scale = delta_p / g = 30000/9.80665 ≈ 3060.5  kg/m²  per (kg/kg)
div_mm_s = scale * div  → mm/s  (since 1 kg/m² = 1 mm)
dpwv_dt : mm / s
P_mm_s  : rain[mm/h] / 3600 → mm/s
residual = dpwv_dt + div_mm_s + P_mm_s  → all mm/s ✓
```
注释 L1-27 把单位推导写在 module docstring 顶部,审核友好。

### 1.3 符号约定 — 正确
真方程: `dPWV/dt = -∇·(qV) - P + E`
代码 residual = `dPWV/dt + ∇·(qV) + P`,等价于 `-E`(蒸发项被吸收入残差)。
对短时尺度(6 min cadence)E ≈ 0,作为软约束最小化 |residual| 物理合理。

注意:`compute_mfd.py` 返回的是 **MFD = -div(qV)**(收敛为正),而本模块直接使用 div(qV)(辐合为负),符号差是有意的——`scale*div(qV) = -scale*MFD`,叠加到 residual 中数学等价。审核员复核了 numpy 输出的统计(L97-98 attrs `sign_convention=positive means convergence`),与本 PyTorch 版无矛盾。

## 2. 可微性 — 通过

### 2.1 实现层面
- 全程 `F.pad + slicing + torch.cat`,**无原地写操作**(无 `tensor[...] = ...`),与 docstring L26 承诺一致
- Gaussian smoother 用 `F.conv2d`,kernel 注册为 buffer
- 输入 `qu = q*u`,`qv = q*v`(L278-279)经乘法/链式法则保持 grad

### 2.2 实测(Test 2)
跑通 `loss.backward()` 后:
```
|grad_pwv|.sum() = 3.258e-03   (≠0)
|grad_rain|.sum() = 2.726e-04  (≠0)
```
梯度成功传到 `pwv_pred` 和 `rain_pred`,**通过**。

## 3. 数值稳定性

| 维度 | 实现 | 评级 |
|---|---|---|
| 除零 | `n = mask.sum().clamp(min=1.0)` (L296),Huber denom `0.5*r²/d` 中 d 常量 >0,FSS `den.clamp(min=1e-12)` | PASS |
| Huber 分支 | abs_r ≤ d 用 quad,否则 lin(L301-306) | PASS |
| Mask 形状校验 | L290-293 严格 shape check | PASS |
| Lat 方向校验 | L154-155 显式 raise | PASS |
| Shape 一致性 | L270-276 一次性校验所有 5 个张量 | PASS |
| 梯度裁剪 hook | **未内置**,但 Huber 已起到隐式保护;集成测试 T4.5 可在 trainer 中 `clip_grad_norm_`。非阻塞。 | OK |

## 4. 与 compute_mfd.py 一致性 — 通过

Test 5 在 `(T,H,W)=(3,21,25)` 随机场上对照:
```
interior max-abs rel error = 9.03e-08
whole-field max-abs rel error = 1.02e-07
```
**远低于 5% 阈值**(float32 精度极限附近)。docstring 自报 "1.2e-7 rel error" 与实测吻合。两版边界单侧差公式逐字对应。

## 5. 测试覆盖与执行 — 全部通过

实跑 `python model/losses/test_water_budget.py` (CPU torch 2.12+cpu):

| # | 测试 | 结果 | 关键数 |
|---|------|------|--------|
| 1 | 平流场 residual~0 | PASS | mean\|res\|/mean\|dPWV/dt\| = 2.08e-2 (<5%) |
| 2 | 可微性 grad→pwv,rain | PASS | grad_pwv=3.26e-3, grad_rain=2.73e-4 |
| 3 | 球面 vs 欧式 @40N | PASS | 比值 1.3060 vs 期望 1/cos(40°)=1.3054, 误差 0.047% |
| 4 | 零输入零损失 | PASS | loss=0 |
| 5 | 与 numpy compute_mfd 一致 | PASS | rel err 9.03e-8 |
| 6 | Gaussian vs scipy | PASS | rel err 1.37e-7 |
| 7 | 时间导数线性场 | PASS | rel err 0 (exact) |

**All water-budget tests passed.**(stdout 末行)

注意:test 通过 `importlib.util.spec_from_file_location` 直接加载 water_budget.py(L28-38),绕开 `model/__init__.py`(避免 T4.1 的 Pluvian import 副作用)——这是合理的隔离设计,但 trainer 集成时要直接 `from model.losses import WaterBudgetLoss`,届时依赖 `model/__init__.py` 的 import 顺序,T4.5 需复核。

## 6. data_losses.py 附加审查 (非任务必需项,顺手过一遍)

- `WeightedMSE`:分段权重 1/1/3/5,denom 用权重 sum 而非像素 count——正确(否则高权样本相对衰减)。
- `BMAE`:wet/dry 各自归一再 0.5 平均,避免类别不平衡偏置。L74-75 双 clamp(min=1.0) 安全。
- `FSSProxy`:用 sigmoid 替代硬阈值实现可微,`avg_pool2d count_include_pad=False`(L120)是关键正确细节(否则边界 box 平均偏小)。变量命名 `fss = 1.0 - num/den` 然后 `return 1.0 - fss` 略绕,但等价于直接 `return num/den`,功能正确。
- `FocalRainLoss`:p 双侧 clamp(L163)防 log(0),α/γ 标准实现。

## 加分项
1. Module docstring 顶部把 PWV/dPWV/dt/div/P 的单位推导列出来,审核效率极高
2. `delta_p_pa=30000` 默认值有注释解释("700-1000 hPa column proxy")
3. Test 1 用解析平流场(高斯包东移)做闭式验证,残差应为 0,验证了 d/dt 与 div(qV) 的联合正确性,而不是孤立单元测试
4. Test 5 在 test 文件内 inline 复制 numpy 参考实现(L208-225),避免 import xarray 重依赖,可移植性好
5. 默认 `smooth_sigma=2.0` 与 `compute_mfd.py:SIGMA=2.0` 一致,行为可对齐
6. 边界单侧 FD 写法与 numpy 版逐字对应,降低后期漂移风险

## 小建议(非阻塞)

1. **梯度裁剪 hook**:WaterBudgetLoss 自身无 clip,但收支残差对 wind/q 的尾部(强对流)可能产生大梯度。建议 T4.3 trainer 中加 `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`,在 audit_T4_3 中复核。
2. **TimeDerivative.dim 默认 -3**:对 `(B,T,H,W)` 是正确的,但调用方若传入 `(B,C,T,H,W)` 会错位。建议在 docstring 加一行 "expects time at dim=-3 (i.e. shape ends with (T,H,W))",或在 forward 内 assert `x.dim() == 4`。当前 WaterBudgetLoss.forward 已 enforce dim()==4(L275),所以经由 WaterBudgetLoss 调用是安全的。
3. **delta_p 物理性**:30000 Pa 假设 700-1000 hPa 整层 q≈q(850),在湿对流强切变下偏差较大。建议 RESEARCH_PROPOSAL_v1 的 ablation 表里加一组 `delta_p_pa ∈ {20000, 30000, 50000}` 敏感度。非本审核范围,T4.3 config 设计时考虑。
4. **FSSProxy 命名**:可直接写 `return num/den`,去掉中间变量,但不影响正确性。

## 结论

**PASS** — 物理正确、单位严谨、可微性已验证、与 numpy 参考版一致到机器精度、测试 7/7 通过、零硬编码风险。可作为 T4.3 trainer 与 T4.5 集成测试的稳定基线。

下游(T4.3 ablation 5: `budget_loss_only`)可直接 import 使用。
