# Band Debate — 辩手 C 论证

**主张**：7 bands maximum coverage
- `band_centers = (0, 0.5, 2.0, 8.0, 30.0, 80.0, 200.0)`
- `band_sample_weights = (0.05, 0.3, 1.0, 3.0, 12.0, 40.0, 100.0)`
- `band_edges = (0.1, 1.0, 4.0, 16.0, 50.0, 130.0)`

---

## 1. 核心论点：信息容量是这次重训的唯一硬约束

旧 5-band `(0, 0.5, 4.5, 19, 50)` 的失败不是"训练不够"，是**物理上限**——softmax 的期望永远 ≤ 50 mm/h，target max 413，差了 8 倍。这是 ckpt 直接死掉、必须重训的根因。

重训既然要付 60 epoch 的全部成本，**头部容量是边际免费的**：5→7 只多 28 params per spatial cell × O(H·W)，相比 backbone 是噪声级别。真正的 trade-off 不是参数量，是 **"信息容量 vs 训练信号稀疏性"**——这正是 C 相对 A/B 的优势区。

C 给了 7 个 bins 覆盖 [0, 200+]，A/B 顶多 6 bins 顶到 100。把"413 mm/h 的极端事件"硬塞进顶 bin = 100，等于复刻旧 ckpts 的物理上限问题，只是阈值从 50 抬到 100——治标不治本。

## 2. Bands 跟 CSI 阈值的精确对齐（数学论证）

CSI 评估阈值是 `(1, 5, 10, 30)` mm/h。Softmax 期望 `E[y] = Σ p_i · c_i`，要让 model 在阈值处有"crisp decision"，**band centers 应当夹住阈值**，且阈值附近 bin 间距越小越好（梯度信号更密集）。

| CSI 阈值 | C 的覆盖（左/右 band center） | 阈值与最近 center 距离 |
|---:|---|---:|
| 1 mm/h | 0.5 / 2.0 | 0.5 |
| 5 mm/h | 2.0 / 8.0 | 3.0 |
| 10 mm/h | 8.0 / 30.0 | 2.0 |
| 30 mm/h | 30.0（命中）/ 80.0 | **0** |

**关键**：CSI@30 阈值正好命中 band center 30——softmax 只要 mass 集中在该 bin，期望就精确等于 30，没有插值误差。CSI@10 被 8.0 和 30.0 夹住，期望从 8 抬到 10 只需要 ~9% mass 转到 30 bin，这是 model 容易学的精度。

旧 5-band 在 `(4.5, 19)` 之间根本没有 bin，10 这个阈值只能靠 4.5→19 的插值，softmax 一稍偏就跳过 10——这是旧 ckpts CSI@10 烂的几何原因。

## 3. Light rain 区也没省事——4 bands 覆盖 0-10

target 分布 11% 像素 >1 mm/h，是绝大多数样本的工作区。C 在 [0, 10] 给了 `(0, 0.5, 2.0, 8.0)` 4 个 bands，比 A/B 的 3 bands 多一档 fine resolution。p90 = 1.24 落在 0.5 和 2.0 之间，p95 = 3.02 落在 2.0 和 8.0 之间——**90-95 分位的样本被 4 个 bands 精确包夹**，这正是 baseline 性能的票仓。

## 4. Extreme tail 不是浪费——是唯一能打破天花板的杠杆

>50 mm/h 只占 0.009% 像素，乍看 80 / 200 两个 bin 训练信号极稀疏。但这恰恰是 sample_weights 40 / 100 的用武之地：

- 旧 ckpts 顶 band 50 + weight 8 = 总有效信号 `0.009% × 8 = 0.072%` → model 学不会
- C 顶 band 80 + weight 40 = `~0.01% × 40 = 0.4%` → 5.5× 提升
- C 极端 band 200 + weight 100 = `~0.0003% × 100 = 0.03%` → 弱但非零

**没有 weight 100 的极端加权，model 一定会塌缩到 "永远不预测 band 7"**——这是必须接受的成本，不是 bug。

## 5. 主动承认弱点 + 为什么仍 OK

| 弱点 | 严重度 | 为什么仍 OK |
|---|---|---|
| 28 params/cell 新增，60 epoch 可能 underfit head | 中 | head 是 1x1 conv 维度，收敛极快；underfit 风险落在 backbone，跟方案无关 |
| sample_weight 100 早期 loss 爆炸 | 中-高 | 用 5-epoch linear warmup `weight × min(epoch/5, 1)` 解决 |
| 跟旧 dBZ-era 5 bands 不可比，paper rewrite scope 大 | 高 | 旧 ckpts 已死，rewrite 不可避免；A/B 也得 rewrite，C 不更亏 |
| 顶 band 200 几乎 zero signal，可能学 garbage | 中 | 即便 band 7 学不会，bottom 6 bands 已优于 A/B；下行风险被 floor 住 |

## 6. Vs A/B 的关键差分

- **vs A（6 bands 顶到 100）**：A 顶 bin = 100 仍低于 target max 413，复刻天花板问题；C 顶 bin 200 才进入"覆盖 99.99% 累积质量"区
- **vs B（保守 weights）**：B 解决了 bin 位置但不解决 head 学不到 extreme 的问题；C 用 weight 100 强行注入信号

## 7. Mitigation（一句话 actionable）

**对 `band_sample_weights` 做 5-epoch linear warmup，并在 val loss 上对 band 6/7 单独 log CSI@30/CSI@50，前 10 epoch 若 band 7 prediction 占比 < 0.0001% 就触发额外 oversample**——这把"weight 100 不稳定"和"顶 bin 学不会"两个弱点同时 hedge 住。

---

## Vote: C

理由一句话：**A/B 都没有真正解决"head 物理上限"的根因**，只是把上限从 50 抬到 100；只有 C 的顶 bin 200 + weight 100 在数学上能让 softmax 期望覆盖 target 的 [0, 200+] 区间，同时 4 个 light bands 跟 CSI@1/5/10 精确夹住，CSI@30 命中 band center。head 容量在 60 epoch 重训预算下是边际免费，weight warmup 解决稳定性——**不选 C 等于明知道天花板还故意留着**。
