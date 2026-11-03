# Simulated-mission evaluation (4.3) and ablations (7.2)

Source: simulator. Seeds [7], 4 missions each. Synthetic site: rows of near-identical solar panels, a pole line, a wall, machinery; damage grows between missions. Backbone: weight-free projection embedder, so absolute numbers are a lower bound for DINOv2; the comparison between arms is the point.

## Re-identification

| Arm | Rank-1 | False-new rate | False-match rate | Revisits |
|---|---|---|---|---|
| appearance | 0.800 | 0.150 | 0.050 | 60 |
| location + appearance | 0.933 | 0.017 | 0.050 | 60 |
| location + appearance + sift | 0.933 | 0.050 | 0.017 | 60 |

## Change detection

57 revisit pairs, 22 with real damage growth. Mask AUC 0.938.

| Detector | Precision | Recall | F1 | False changes / 100 unchanged |
|---|---|---|---|---|
| label difference | 1.000 | 0.091 | 0.167 | 0.0 |
| patch difference (b ≥ 0.3) | 0.773 | 0.773 | 0.773 | 14.3 |
| pixel mask (≥ 1%) | 0.889 | 0.727 | 0.800 | 5.7 |
| engine M-gate | 0.714 | 0.227 | 0.345 | 5.7 |

## Forecast

8 damaged assets, 8 with tracked defect instances.

| Method | MAE (days) to critical | Series |
|---|---|---|
| severity slope | 31.4 | 1 |
| growth model (2.3) | 57.4 | 2 |
