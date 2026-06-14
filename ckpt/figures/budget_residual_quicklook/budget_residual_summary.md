# Pluvian Water-Budget Residual Diagnostics

This is a physical-consistency diagnostic, not a forecast-skill metric. Lower absolute residuals mean the predicted PWV tendency, ERA5 moisture-flux proxy, and predicted precipitation are more mutually consistent under the configured water-budget proxy.

| model | split | windows | days | mean | rmse | p95 abs | budget loss | rain mm/h | abs dPWV/dt | abs div(qV) | abs P |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ab2 | event_test | 2 | 1 | 0.002763 | 0.003901 | 0.008519 | 0.002713 | 9.606112 | 0.000077 | 0.000387 | 0.002668 |
| ab2 | test_robust | 2 | 1 | 0.000867 | 0.001551 | 0.003653 | 0.000820 | 2.431763 | 0.000068 | 0.000375 | 0.000675 |
| ab3 | event_test | 2 | 1 | 0.002594 | 0.003790 | 0.008502 | 0.002544 | 8.978138 | 0.000072 | 0.000387 | 0.002494 |
| ab3 | test_robust | 2 | 1 | 0.000814 | 0.001508 | 0.003596 | 0.000767 | 2.132823 | 0.000059 | 0.000375 | 0.000593 |

## Residual by Predicted Rain-Rate Bin

Values are mean absolute residual in mm/s, binned by predicted rain rate in mm/h on the ERA5/radar overlap grid.

| model | split | 0-1 | 1-5 | 5-10 | 10-30 | >=30 |
|---|---|---:|---:|---:|---:|---:|
| ab2 | event_test | 0.000368 | 0.000928 | 0.002206 | 0.004997 | 0.009262 |
| ab2 | test_robust | 0.000325 | 0.000708 | 0.002199 | 0.004222 | 0.009551 |
| ab3 | event_test | 0.000354 | 0.000918 | 0.002124 | 0.005009 | 0.009250 |
| ab3 | test_robust | 0.000330 | 0.000783 | 0.002186 | 0.004474 | 0.009548 |
