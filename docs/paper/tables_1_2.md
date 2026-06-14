# Pluvian Paper Tables 1 and 2

Headline deterministic skill scores for the four model variants on the two evaluation splits. CRPS is omitted because the deterministic single-member CRPS reduces to MAE (see `_crps_marginal` in `scripts/train.py`).

### Table 1. Deterministic skill on event_test (4 storm days)

| Model | CSI@1 | CSI@5 | CSI@10 | CSI@30 | FSS@3 | FSS@11 | MAE |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ab1 (radar only) | 0.397 [0.237, 0.494] | 0.508 [0.325, 0.604] | 0.565 [0.389, 0.643] | **0.308 [0.267, 0.345]** | 0.579 [0.393, 0.672] | 0.593 [0.407, 0.684] | 5.188 [3.624, 6.113] |
| ab2 (+PWV via xcoreA) | 0.432 [0.253, 0.543] | 0.535 [0.337, 0.641] | 0.591 [0.391, 0.684] | 0.283 [0.255, 0.307] | 0.614 [0.415, 0.714] | 0.630 [0.431, 0.728] | 4.827 [3.386, 5.667] |
| ab3 (+ERA5) | 0.470 [0.290, 0.572] | 0.570 [0.391, 0.653] | 0.632 [0.461, 0.701] | 0.291 [0.264, 0.327] | 0.651 [0.463, 0.739] | 0.670 [0.484, 0.754] | 4.265 [2.809, 5.195] |
| ab3b (+water-budget loss) | **0.492 [0.301, 0.602]** | **0.582 [0.399, 0.667]** | **0.641 [0.476, 0.706]** | *0.210 [0.188, 0.270] (down)* | **0.672 [0.476, 0.762]** | **0.691 [0.499, 0.779]** | **4.098 [2.744, 5.060]** |

_Split: event_test (4 days, 96 forecast windows). CSI thresholds in mm h$^{-1}$; FSS neighbourhood in grid pixels (1 pixel = 0.01$^\circ$ $\approx$ 1.1 km). MAE in mm h$^{-1}$. Brackets show day-stratified bootstrap 95% CI (n_boot = 1000). Bold marks the best value per column; ab3b CSI@30 is italicised with a down arrow to flag the inverse direction of its regression._

### Table 2. Deterministic skill on test_robust (non-storm days)

| Model | CSI@1 | CSI@5 | CSI@10 | CSI@30 | FSS@3 | FSS@11 | MAE |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ab1 (radar only) | 0.324 [0.218, 0.399] | 0.437 [0.315, 0.510] | 0.495 [0.365, 0.562] | 0.270 [0.216, 0.297] | 0.502 [0.369, 0.583] | 0.520 [0.387, 0.600] | 3.035 [2.118, 4.070] |
| ab2 (+PWV via xcoreA) | 0.330 [0.222, 0.405] | 0.443 [0.324, 0.512] | 0.505 [0.377, 0.564] | 0.273 [0.219, 0.290] | 0.508 [0.375, 0.588] | 0.527 [0.393, 0.606] | 2.948 [2.021, 3.980] |
| ab3 (+ERA5) | 0.379 [0.268, 0.453] | 0.495 [0.374, 0.562] | 0.549 [0.416, 0.607] | **0.276 [0.236, 0.292]** | 0.565 [0.438, 0.638] | 0.588 [0.462, 0.659] | 2.475 [1.629, 3.435] |
| ab3b (+water-budget loss) | **0.382 [0.259, 0.467]** | **0.500 [0.370, 0.575]** | **0.558 [0.421, 0.618]** | *0.215 [0.178, 0.244] (down)* | **0.568 [0.426, 0.652]** | **0.591 [0.449, 0.674]** | **2.409 [1.633, 3.299]** |

_Split: test_robust (8 days, 192 forecast windows). CSI thresholds in mm h$^{-1}$; FSS neighbourhood in grid pixels (1 pixel = 0.01$^\circ$ $\approx$ 1.1 km). MAE in mm h$^{-1}$. Brackets show day-stratified bootstrap 95% CI (n_boot = 1000). Bold marks the best value per column; ab3b CSI@30 is italicised with a down arrow to flag the inverse direction of its regression._
