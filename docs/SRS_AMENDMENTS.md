# SRS amendments

Changes the AUDI SRS document itself needs so that it matches the implemented and
tested system. Section numbers refer to the SRS. Every value below lives in
`config.yaml` and is the value the tests and benchmarks ran with.

## 1. §9 traceability matrix and §11.4 increment table

The §9 matrix assigns several requirements to the wrong module group. Corrected:

| Module group | Requirements |
|---|---|
| Memory Management | FR-1 – FR-7 |
| Perception & Matching | FR-8 – FR-12 |
| Change Detection & Alerting | FR-13 – FR-17 |
| Reporting & Escalation | FR-18, FR-19 |
| Predictive Revisit | FR-20 |

The §11.4 increment table must use the same grouping, so that each increment
delivers whole module groups:

| Increment | Delivers |
|---|---|
| 1 — Memory core | FR-1 – FR-7 (local DB, cloud pull, offline mode, sessions, persistence, sync, outbox) |
| 2 — Perception | FR-8 – FR-12 (detection, embedding, identity cascade, new-asset registration, history retrieval) |
| 3 — Change & alerting | FR-13 – FR-17 (change magnitude, severity, live alerts, confirmed path, defect lifecycle) |
| 4 — Reporting | FR-18, FR-19 (post-flight report, escalation) |
| 5 — Revisit | FR-20 (priority ranking in the pre-flight briefing) |

The requirement → module → test mapping for this code base is in
`docs/SRS_TRACEABILITY.md` and is enforced by `tests/test_srs_traceability.py`.

## 2. §2.5 and §5.4: offline-mode references

§2.5 (operating environment) and §5.4 (connectivity) cite FR-17 for offline
operation. FR-17 is the defect lifecycle. Offline operation is specified by:

- **FR-3**: if the cloud is unreachable at session start, the session starts in
  offline mode from the local DB.
- **FR-7**: local writes are queued and synchronised when connectivity returns.

Both sections should cite FR-3 and FR-7.

## 3. Numeric targets for NFR-1 – NFR-3

The SRS states these NFRs qualitatively. Measurable targets:

| ID | Requirement | Target | Measured by |
|---|---|---|---|
| NFR-1 | Alert latency, from frame capture to the alert being available to the overlay | ≤ 500 ms p95 on the development laptop | `scripts/bench.py` (recorded video through the live pipeline) |
| NFR-2 | Asset matching per detection with 5,000 assets in memory | ≤ 50 ms p95 | `scripts/bench.py --assets 5000` |
| NFR-3 | Start/end sync must not delay take-off | ≤ 10 s, then the session proceeds offline | `scripts/bench.py` (cloud down, black-holed, slow, healthy) |

NFR-1 is measured from the timestamp the camera reader assigns when a frame is
grabbed, not from when processing starts, so time spent queued counts towards
the latency.

## 4. §8.1 severity: confidence term and change gate

`S = (w1·M + w2·C)·Cf` with `w1 = 0.6`, `w2 = 0.4`.

- **Cf is observation confidence**: `Cf = min(detector confidence,
  condition-classifier confidence)`. It is not the identity-match score. A
  confident match to a badly damaged asset must not reduce that asset's
  severity, and a weak detection must not produce a high-severity alert.
- **M gate**: if `M < change_detection.threshold` (0.30), the change is not
  significant, so `S = 0` and the asset follows the FR-16 "confirmed" path. Without
  the gate, every unchanged asset gets a severity floor of `w2·C·Cf`, which is
  0.28–0.36 for the configured criticalities. That would make severity
  meaningless as an alert signal.
- Levels (lower bounds): low 0.0, medium 0.3, priority 0.5, critical 0.8. The
  live overlay beeps at priority and shows a red border at critical.

## 5. §8.2 revisit priority: trend window and time term

`R = w3·Sr + w4·T + w5·D` with `w3 = 0.5`, `w4 = 0.3`, `w5 = 0.2`.

- **T** is computed over the last **N = 5** missions
  (`revisit_priority.trend_window`). It is the least-squares slope of severity
  against mission index, multiplied by `(n − 1)` (the fitted rise across the
  window) and clipped to [0, 1]. A flat or falling trend gives 0.
- **D** = `min(1, days since last inspection / 30)`. It is computed **when the
  briefing is requested**, not stored when the asset was last observed, so it
  keeps growing for assets that are not revisited. R is recomputed for every
  asset at the site, including assets not seen in the latest mission.

## 6. New interfaces

§3 (external interfaces) needs two additional inputs:

- **Live camera / stream input.** Sources can be a local camera (USB webcam by
  index or device name), an RTSP/HTTP stream URL, or a recorded video file. The
  camera reader reconnects automatically after a disconnect and always hands
  the pipeline the most recent frame. The browser camera can be used as a
  fallback (frames sent over a WebSocket). An annotated MJPEG stream and a
  WebSocket event feed drive the operator overlay, which is drawn as edge bars
  so the centre of the view stays unobstructed (FR-15, §6.3).
- **ArUco-tagged assets.** An asset may carry a printed ArUco marker
  (dictionary `DICT_4X4_50`). A marker decoded inside the detection box is the
  first, authoritative step of the identity cascade. The cascade is ArUco
  marker, then location gate, then top-k embedding retrieval, then SIFT/RANSAC
  geometric verification, then visual-only match, then new asset. Unmarked
  assets fall through to the visual steps.
