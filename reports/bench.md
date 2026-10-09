# AUDI benchmarks (NFR-1/2/3)

Generated 2026-10-08T21:21:12.207539+00:00 on Darwin arm64 · Python 3.11.14 · device `mps` · backbone `dinov2_vits14`.

| Requirement | Measured | Target | Result |
|---|---|---|---|
| NFR-1 alert latency p95 | 127.4 ms (p50 98.3, n=12) | ≤ 500 ms | PASS |
| NFR-2 matching p95 @ 5000 assets, after session-start prefetch | 32.3 ms (p50 9.7) | ≤ 50 ms | PASS |
| NFR-2 matching p95 @ 5000 assets, cold cache | 32.4 ms (p50 12.4) | ≤ 50 ms | PASS |
| NFR-2 matching p95 @ 5000 assets, warm cache | 17.1 ms (p50 4.7) | ≤ 50 ms | PASS |
| NFR-3 worst start/end sync | 10.03 s | ≤ 10 s (+1 s slack) | PASS |

## NFR-1 — live pipeline

Input: recorded video: 12 building assets, second flight with synthetic cracks. Capture FPS 15.0; alerts raised: 12 (critical, priority); defects flagged in the post-flight report: 12.

Counters: `{"seen": 359, "processed": 12, "skipped_frame_skip": 180, "skipped_duplicate": 167, "events": 12, "alerts": 12, "errors": 0}`

Event types: `{"new_asset": 8, "change": 4}`. On the recorded video every asset was stored in the first flight, so `change` = re-identified with its damage detected and `new_asset` = re-identification failed (the asset is registered again; it still alerts because the classifier sees the crack).

| Stage | p50 ms | p95 ms | n |
|---|---|---|---|
| alert | 98.3 | 127.4 | 12 |
| capture_to_result | 98.2 | 127.3 | 12 |
| change | 0.0 | 31.8 | 12 |
| classify | 0.3 | 0.4 | 12 |
| detect | 0.0 | 0.0 | 12 |
| embed_batch | 43.8 | 49.0 | 12 |
| frame_total | 95.9 | 125.2 | 12 |
| identity | 39.9 | 50.0 | 12 |
| store | 3.7 | 8.9 | 12 |
| total | 50.5 | 81.8 | 12 |

## NFR-2 — identity cascade

5000 assets / 5000 prototypes in one partition, `numpy` backend; queries are webcam-augmented re-photographs of stored assets.

| Pass | p50 ms | p95 ms | index search p95 ms | top-1 | false new | stages |
|---|---|---|---|---|---|---|
| cold: no reference keypoints cached | 12.4 | 32.4 | 0.23 | 0.845 | 0.130 | `{"geometric": 168, "new": 26, "visual_only": 6}` |
| after session-start prefetch (2048 of 5000 references, 42% of queried assets covered) | 9.7 | 32.3 | 0.23 | 0.845 | 0.130 | `{"geometric": 168, "new": 26, "visual_only": 6}` |
| warm: every queried reference cached | 4.7 | 17.1 | 0.21 | 0.845 | 0.130 | `{"geometric": 168, "new": 26, "visual_only": 6}` |

The session-start prefetch (`SemanticMemoryEngine.prefetch_references`, run in the background by `MissionService.start`) took 11.3 s for 2048 references (`matching.feature_cache_size`).
Embedding the query crop is part of NFR-1 (the `embed` stage in the live pipeline), not matching.

## NFR-3 — sync bound

`cloud.sync_timeout_s` = 10.0

| Scenario | start s | end s | session mode | start timed out | end online |
|---|---|---|---|---|---|
| cloud down (connection refused) | 0.02 | 0.01 | offline | False | False |
| cloud black-holed (no response) | 2.02 | 2.02 | offline | False | False |
| cloud slow (pull/push stall 30 s) | 10.03 | 10.03 | offline | True | False |
| cloud healthy | 0.22 | 0.01 | online | False | True |
