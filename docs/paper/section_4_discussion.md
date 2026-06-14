# §4 Discussion (draft 2026-06-14)

### 4.1 Summary of findings

Three results structure the remainder of this discussion. Stepwise multimodal fusion (radar → +PWV → +ERA5) improves light-to-moderate skill monotonically (CSI@1–10, FSS, MAE) but leaves CSI@30 essentially unchanged on event_test, with ab3 (0.291) failing to overtake the radar-only baseline ab1 (0.308). Adding a column water-budget PDE term (ab3b) extends the bulk-metric gains yet collapses CSI@30 by 28% (event_test) and 22% (test_robust). A pooled-CSI diagnostic separates the two regimes: ab3 recovers tail skill under enlarged tolerance, ab3b does not.

### 4.2 Why a physical loss can hurt the extreme tail

The water-budget residual penalises, at every pixel and step, the discrepancy between predicted precipitation, the local PWV tendency, and the surface moisture flux. The constraint is locally pointwise and quadratic, which has a specific geometric consequence: minimising such a residual under noisy reanalysis-grid inputs favours spatially smoother solutions, because the PWV time derivative over convective cores is poorly resolved on the ERA5 / GNSS network and contributes the largest residual when the predicted precipitation field is sharply peaked. The optimiser therefore reduces the residual most cheaply by attenuating the peaks rather than by sharpening the moisture fields it cannot modify. This is consistent with the ab3b regression being concentrated exclusively at the high-reflectivity tail (CSI@30, 0/18 leads positive) while the bulk metrics continue to improve.

This contradicts a common framing in which physics-informed losses are reported as uniformly beneficial. The relevant question is not whether a constraint is physically motivated but what kind of solution its loss landscape prefers. A pointwise quadratic penalty on a noisy derivative is structurally biased towards the smooth solution; in a nowcasting context that bias destroys exactly the high-intensity tail the application cares about. Plausible remedies preserve the physical intent while changing the geometry: a weak-form (integral) budget over storm-scale subdomains relaxes the pixelwise pressure; an adversarial term [Ravuri 2021] supplies a counter-gradient against blurry solutions; and enforcing the constraint in a latent representation rather than on the raw precipitation field decouples sharpness from conservation. We do not adjudicate among these here but flag the design choice as a first-order one.

### 4.3 Pooled-CSI as a community diagnostic

The pooled-CSI sweep is a one-parameter extension of the standard categorical score that, at no architectural or annotation cost, separates two failure modes that an unpooled CSI conflates: a displacement error in which the correct core is placed in the wrong location, and a physical-smoothing error in which the core has been dissolved. Without this separation, ablation tables comparing a physics-augmented model to its baseline cannot honestly report whether a CSI@30 regression reflects a tolerable spatial offset or an intrinsic loss of intensity. In our case the answer is unambiguous: ab3b's pooled-CSI plateau at w=16 (0.222) remains below ab3 at w=1 (0.291).

The role we propose is analogous to that of the Fractions Skill Score [Roberts & Lean 2008] in numerical weather prediction, where neighbourhood evaluation became standard precisely because gridpoint scores penalised acceptable phase errors. Pooled-CSI complements FSS by retaining the intensity threshold and exposing the smoothing/displacement decomposition. Two limitations are worth stating: it is a deterministic diagnostic and cannot substitute for ensemble probabilistic verification (CRPS, reliability), and the pooling neighbourhood is defined in grid units, so cross-study comparison requires fixing a physical scale.

### 4.4 Architecture as the remaining lever

If reweighted losses and hard physical constraints both have undesirable side effects on the extreme tail, the remaining intervention is at the architecture level. The CDU dual-branch decoder [Yang et al. 2026] is motivated by the observation that a shared upsampling path low-pass-filters the high-frequency component of the precipitation field; routing high- and low-frequency signals through separate branches preserves intense cores while retaining context. Our ab3-cdu configuration tests this hypothesis on the same data and protocol as ab1–ab3b (§3.4, pending). Independent of that outcome, pooled-CSI gives us the means to interpret it: if CDU recovers CSI@30, the diagnostic will show that recovery comes from intensity preservation rather than displacement reduction, and if it does not, the diagnostic will localise the residual failure mode.

### 4.5 Limitations and future work

Five caveats bound the scope of these claims. (a) Evaluation samples are small: event_test covers four storm days and test_robust eight days; multi-season, multi-region extension is necessary before the smoothing-vs-displacement finding can be claimed as general. (b) The model is deterministic and single-member, so MAE and CRPS collapse to the same quantity and we cannot discuss ensemble spread or probabilistic calibration; ensembling is a clean next step. (c) GNSS-derived PWV is dense over North China but sparser elsewhere; transfer to western China or maritime regions will require reassessment of the interpolation error budget. (d) ERA5 is a reanalysis, not an operational product; replacement by IFS HRES or GFS analyses at deployment time will introduce a performance gap we have not quantified. (e) Training data are restricted to May–August 2023, leaving cold-season stratiform and mixed-phase regimes outside the validated envelope.

<!-- editorial notes
1. 全文约 820 words，§4.2 略偏长（≈260），因为机制讨论是论文最有 originality 的论点之一，建议保留。如需压到 800，可砍 §4.5 最后一句关于 cold-season 的展开，或合并 (c)(d) 两条。
2. §4.4 严格遵守 "CDU 必须写 pending" 的硬约束；现写法把 pooled-CSI 作为"无论 CDU 成败都有诊断价值"的兜底论点，避免论文叙事被实验进度卡死。
3. §4.2 提出的三个 remedy（weak-form / adversarial / latent-space）只点到为止、未承诺工作量；与 §4.5 的 future work 不重复以免读者觉得 padding。
4. 引用全部用 [Author Year] 占位，待 §5 References 统一时再回填具体卷期。已用占位：Ravuri 2021（DGMR）、Roberts & Lean 2008（FSS）、exPreCast（CDU 原始论文）。
5. §4.1 故意写得短而硬，让读者带着结论进入后续机制讨论；与 §3 末尾的 pooled-CSI 总结自然衔接。
-->
