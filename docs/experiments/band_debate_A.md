# Band Debate — 辩手 A: Minimal Extension to 6 Bands

**Proposal A**
- `band_centers      = (0, 0.5, 4.5, 19, 50, 150)`  # +1 band
- `band_sample_weights = (0.1, 1, 2, 4, 8, 16)`     # +1 weight
- `band_edges        = (0.1, 1, 8, 30, 80)`         # +1 edge

## 1. 为什么仍是最优选

**核心论点：在「修复 50 mm/h convex cap」这件唯一必须做的事上，A 的改动量最小、风险最低、可比性最强。**

当前 head 把预测压到 `softmax @ centers`，5 个 ckpt 全部物理上限 50 mm/h。target max 413、p99.99=49、p99.9=24，>50 mm/h 像素占 0.009%。也就是说，**任何修复方案都必须把顶 center 推到 ≥150 才能在数学上覆盖 p99.99 之上的尾部**——否则 CSI@30 在极端事件上永远偏低（因为模型连「猜对量级」的能力都没有）。A 直接把顶 center 设到 150，覆盖到 target max 413 的 ~36%，且因为 >150 mm/h 像素 ≈ 0%，cap-at-150 在 test 集上几乎不会触发。

**对比其他思路**（替方案 B/C/D 预辩护）：
- 「重新均匀分箱（如 0/2/8/30/80/200）」：所有 band 都动，5 个 ckpt 的 logits 几何意义全部失效，strict=False 加载等于 head 从 0 训，前面 6 个 backbone block 的归纳偏置浪费。A 保留前 5 个 center 不变，head 的 0–50 mm/h 区间仍可热启动。
- 「保留 5 bands 改顶 center 为 150」：不增维度，但 19→150 跨度 8×，比 A 的 50→150 跨度 3× 更糟，且把原本承担「中雨」的 50-band 直接挪掉，破坏 CSI@30 的 calibration。
- 「上 10+ bands 精细化」：sample_weights 调参空间爆炸，paper 叙事多一段 ablation 债，且高 band 样本数仍受物理限制，精细化解决不了「样本根本不够」的问题。

A 是「**只动必须动的**」——外科手术式修改，符合简单优先原则。

## 2. Target distribution 的数学论证

| 量 | 值 | A 的覆盖 |
|---|---|---|
| target max | 413 mm/h | 150 cap，>150 像素 ≈ 0% (1e-6 级)，几乎不丢信息 |
| p99.99 | 49 mm/h | 落在 50-band 内，原 calibration 保留 |
| p99.9 | 24 mm/h | 落在 19-band 与 50-band 之间，bucketize edge=30 决定 |
| p99 | 8.95 mm/h | 落在 4.5-band 内，原 weight=2 保留 |
| >30 mm/h 像素 | 0.05% | 顶两 band 总样本量 ≈ 5e-4，加权后 effective weight ≈ 0.05% × 8 + 0.009% × 16 ≈ 0.55%，可训 |
| >50 mm/h 像素 | 0.009% | 顶 band 独占，加权 0.009% × 16 = 0.14% effective，**临界可训** |

**结论：A 的 weights schedule (16 on 顶 band) 让 50–150 mm/h 区间的有效梯度信号 ≈ 0.14%，与 phase 7e 时 50-band 在 8× weight 下的 effective 信号 (0.009% × 8 = 0.07%) 在同一量级**。换句话说，模型「能学到顶 band」的难度跟之前学 50-band 的难度差不多——既然 phase 7e 的 50-band 没退化成 dead band，A 的 150-band 也不会。

## 3. 跨 era 可比性 (paper narrative)

Phase 7 论文叙事是「dBZ → mm/h era 重训，head 沿用 5-band 经典 schedule」。A 的 6-band 是「**经典 5-band + 1 个 extension**」，在 method section 可以一句话写完：

> "We extend the canonical 5-band head with one additional band at 150 mm/h to relax the convex upper bound of 50 mm/h, leaving the lower 5 bands unchanged."

而方案 B/C 需要解释「为什么整套 schedule 都重设」，reviewer 会问「你之前 5 ckpt 的 ablation 数据还能用吗？」——答不上来。A 直接说「lower 4 bands schedule 完全一致，对比公平」。这是**唯一能让 phase 7d/7e 已发表的 ablation 表格继续有意义**的方案。

## 4. 必须诚实承认的 weakness

1. **顶 band 几乎不会被激活**：target >50 mm/h 像素 0.009%，>100 mm/h 像素 0.0003%。即使 weight=16，模型大概率会学到 "argmax(softmax) 永不指向顶 band"，因为 cross-band confusion penalty 远超孤注一掷 predict 150 的收益。
2. **50 → 150 跨度 3×，softmax × center 在 50–150 之间无 fine resolution**：模型只能输出 50（顶 band prob=0）或 ~100（顶 band prob=0.5），中间值出不来。CSI@30 受益，但 RMSE on extreme pixels 改善有限。
3. **本质上是 "把 cap 从 50 换到 150"，不是真正的 distribution-aware modeling**：这不是 fundamental fix。

**为什么仍 OK**：
- weakness 1 只影响「最极端 0.009% 像素」的 CSI；CSI@30 这种 paper 主指标的目标阈值 30 mm/h 落在 19-band 与 50-band 之间，A 完全没破坏这块的 calibration。
- weakness 2 在 paper 里不是评估指标——CSI 是阈值二值化，不考核 50-150 之间的连续精度。
- weakness 3 是 phase 7f 的范围之外——「真正 distribution-aware」需要换 loss（如 EVL/GEV head），是另一个 paper 的工作量。

## 5. Mitigation（1 句话）

**训练时对顶 band sample 启用 oversampling（按 patch-level >50 mm/h 像素数加权采样，倍率 5–10×）**，把 effective batch 内顶 band 样本量从 0.009% 提到 ~0.05–0.1%，让 weight=16 真的能更新梯度，避免 dead-band 退化。

---

**vote: A**
