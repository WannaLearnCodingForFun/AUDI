# Real multi-visit dataset — collection protocol (upgrade 7.1)

Public datasets of the **same assets inspected repeatedly** are rare; this protocol produces one.

## Site and schedule

* One real site with 20–60 assets of one family: the campus solar roof, a line of poles, or a building wall.
* 3–5 visits, 1–2 weeks apart, at a similar time of day (±1 h) to limit shadow changes. Record
  weather (sun / overcast / wet) per visit — it is part of the datasheet.
* Same route every time: plan it once with `POST /api/sites/{site}/route-plan?format=plan`
  (QGroundControl) or the drone app's waypoint mission, then re-fly that file.

## Capture

* Camera calibrated once (`scripts/calibrate_camera.py`); keep zoom fixed.
* Log GPS **and** pose per photo: lat, lon, height above ground, heading, gimbal pitch. Export the
  flight log to `filename, latitude, longitude, altitude_agl, heading_deg, pitch_deg` CSV.
* Optional: a tape-measured reference length visible in a few frames (scale ground truth for 3.6),
  ArUco markers on a subset of assets (identity ground truth for free).
* Allowed real changes between visits: natural soiling, bird droppings, snow; staged changes only
  with the owner's permission (chalk "cracks", removable rust decals) and recorded in the change log.

## Package, label, evaluate

```bash
python scripts/multivisit_dataset.py package --visit raw/visit_01 --log raw/visit_01/flight.csv --out release/v1/visit_01
...                                                          # one per visit
python scripts/multivisit_dataset.py prelabel --release release/v1 --site campus_roof
# correct release/v1/labels/visit_XX.csv: asset_id must be the SAME string for the same physical asset in every visit
python scripts/multivisit_dataset.py import --release release/v1
python scripts/run_sim_eval.py --missions-dir release/v1     # re-ID / change tables on real data
```

Label columns: `frame, asset_id, asset_type, x1, y1, x2, y2, defect, defect_extent, changed_since_last_visit, notes`.
Two annotators label one visit independently; report Cohen's κ on `changed_since_last_visit` and
asset-id agreement in the datasheet. Defect masks (for 2.1 / 3.3) are polygons in CVAT exported
as COCO, stored next to the CSVs.

## Release

`release/v1/` = one folder per visit (`frames/`, `frames.jsonl`), `labels/`, `masks/`,
`DATASHEET.md` (fill in `docs/DATASHEET_TEMPLATE.md`), `LICENSE` (CC BY 4.0 suggested) and the
evaluation output. Blur faces / number plates before release; get the site owner's written consent.
