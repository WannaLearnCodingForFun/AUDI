# Simulated-mission evaluation (4.3) and ablations (7.2)

Source: simulator. Seeds [7, 11, 13], 6 missions each. Synthetic site: rows of near-identical solar panels, a pole line, a wall, machinery; damage grows between missions. Backbone: weight-free projection embedder, so absolute numbers are a lower bound for DINOv2; the comparison between arms is the point.

## Re-identification

| Arm | Rank-1 | False-new rate | False-match rate | Revisits |
|---|---|---|---|---|
| appearance | 0.750 | 0.070 | 0.180 | 300 |
| location + appearance | 0.920 | 0.017 | 0.063 | 300 |
| location + appearance + sift | 0.910 | 0.033 | 0.057 | 300 |

## Change detection

290 revisit pairs, 84 with real damage growth. Mask AUC 0.935.

| Detector | Precision | Recall | F1 | False changes / 100 unchanged |
|---|---|---|---|---|
| label difference | 0.938 | 0.185 | 0.412 | 0.4 |
| patch difference (b ≥ 0.3) | 0.579 | 0.826 | 0.671 | 23.4 |
| pixel mask (≥ 1%) | 0.797 | 0.739 | 0.753 | 6.8 |
| engine M-gate | 0.813 | 0.402 | 0.496 | 3.2 |

## Forecast

23 damaged assets, 19 with tracked defect instances.

| Method | MAE (days) to critical | Series |
|---|---|---|
| severity slope | 124.1 | 9 |
| growth model (2.3) | 38.3 | 13 |
