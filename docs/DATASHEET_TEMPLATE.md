# Datasheet: AUDI multi-visit inspection dataset vX

Following Gebru et al., *Datasheets for Datasets* (2021). Fill in every field; write "n/a" rather than deleting.

## Motivation
- Purpose: longitudinal re-identification, change detection and deterioration forecasting of the same physical assets.
- Creators / funding:

## Composition
- Site type and location (coarse, e.g. "university rooftop PV, central Europe"):
- Assets: count by type; assets with ArUco markers:
- Visits: dates, time of day, weather, camera, height above ground, gimbal pitch:
- Frames per visit; image size; GPS / pose fields and their accuracy:
- Labels: asset ids across visits, boxes, defect types and extents, change flags, masks (which visits):
- Changes between visits: natural (soiling, snow) and staged (with the change log):
- Known gaps / errors (missed assets, GPS dropouts):
- Personal data: faces / plates blurred? consent obtained from:

## Collection process
- Route file and drone / camera model; calibration file:
- Who flew and labelled; annotator agreement (κ, asset-id agreement):

## Preprocessing
- `scripts/multivisit_dataset.py` version (git commit):
- Anything removed and why:

## Uses
- Intended: re-ID (Rank-1, false-new rate), change (mask F1, false changes / 100), forecast (error in days).
- Not intended: safety certification of the inspected assets.

## Distribution and maintenance
- Licence: · DOI: · Contact: · Versioning / errata:

## Baselines (from `scripts/run_sim_eval.py --missions-dir`)
Paste the re-ID and change tables here.
