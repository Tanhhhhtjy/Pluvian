# Pluvian Phase 7d Route Summary

Read-only summary from existing eval/per-lead JSON files.

## Aggregate Holdout Metrics

| model | split | status | csi1 | csi5 | csi10 | csi30 | fss3 | fss11 | crps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ablation_2_p7d_droppwv | event_test | ok | 0.4258 | 0.5396 | 0.6004 | 0.1262 | 0.6083 | 0.6240 | 4.9815 |
| ablation_2_p7d_droppwv | test_robust | ok | 0.3269 | 0.4441 | 0.5092 | 0.1518 | 0.5050 | 0.5233 | 2.9683 |
| ablation_2_p7d_xcoreA | event_test | ok | 0.4268 | 0.5275 | 0.5797 | 0.2900 | 0.6093 | 0.6255 | 4.9745 |
| ablation_2_p7d_xcoreA | test_robust | ok | 0.3246 | 0.4339 | 0.4933 | 0.2764 | 0.5024 | 0.5209 | 3.0474 |
| ablation_2_p7d_xcoreB | event_test | ok | 0.4150 | 0.5118 | 0.5614 | 0.3027 | 0.5974 | 0.6130 | 5.2609 |
| ablation_2_p7d_xcoreB | test_robust | ok | 0.3144 | 0.4189 | 0.4776 | 0.2844 | 0.4904 | 0.5082 | 3.2233 |
| ablation_3_era5_p7c | event_test | ok | 0.4696 | 0.5701 | 0.6318 | 0.2907 | 0.6514 | 0.6697 | 4.2649 |
| ablation_3_era5_p7c | test_robust | ok | 0.3789 | 0.4947 | 0.5491 | 0.2755 | 0.5647 | 0.5876 | 2.4749 |
| ablation_3b_era5_budget_p7c | event_test | ok | 0.4917 | 0.5819 | 0.6408 | 0.2099 | 0.6716 | 0.6914 | 4.0978 |
| ablation_3b_era5_budget_p7c | test_robust | ok | 0.3822 | 0.5003 | 0.5585 | 0.2147 | 0.5681 | 0.5913 | 2.4089 |

## Per-Lead CSI30 Delta vs ab1

`csi30_short_delta` averages leads +18..+42 min.

| model | split | status | csi30_mean_delta | csi30_short_delta | min_delta | max_delta | positive_leads |
|---|---|---|---:|---:|---:|---:|---:|
| ab2_p7d | event_test | ok | -0.0259 | -0.0384 | -0.0623 | 0.0054 | 1/18 |
| ab2_p7d | test_robust | ok | 0.0017 | 0.0014 | -0.0090 | 0.0074 | 12/18 |
| xcoreA | event_test | ok | -0.0176 | -0.0267 | -0.0470 | 0.0151 | 2/18 |
| xcoreA | test_robust | ok | 0.0064 | 0.0102 | -0.0014 | 0.0149 | 14/18 |
| xcoreB | event_test | ok | -0.0032 | -0.0099 | -0.0284 | 0.0258 | 8/18 |
| xcoreB | test_robust | ok | 0.0159 | 0.0181 | 0.0084 | 0.0259 | 18/18 |
| ab3_era5 | event_test | ok | -0.0180 | -0.0173 | -0.0439 | 0.0050 | 2/18 |
| ab3_era5 | test_robust | ok | 0.0014 | 0.0051 | -0.0090 | 0.0182 | 8/18 |
| ab3b_budget | event_test | ok | -0.0841 | -0.0940 | -0.0992 | -0.0627 | 0/18 |
| ab3b_budget | test_robust | ok | -0.0631 | -0.0727 | -0.0783 | -0.0492 | 0/18 |

## Route Gate

- ab3 ERA5 holdout status: ready
- ab3b budget holdout status: ready
- Decision rule: do not promote an experiment that fixes csi30 by materially damaging csi1/csi10/FSS/CRPS.
