# AUDI unified dataset

Built by `scripts/download_datasets.py` (raw Kaggle downloads → `dataset/raw/`) and
`scripts/prepare_unified_dataset.py` (→ `dataset/processed/`). Neither folder is committed.

- `dataset/processed/manifest.csv` — one row per image: `path`, `asset_type`, `condition`, `defect_type`, `is_defect`, `source`, `split`, `phash`
- `dataset/processed/classification/<asset_type>/<condition>/` — cropped / resized images
- `dataset/processed/detection/` — YOLO-format asset boxes (`data.yaml`)
- `dataset/processed/STATS.md` — counts, de-duplication, merges and unmapped classes

## Unified conditions

- **solar_panel**: normal, soiling, bird_drop, electrical_damage, physical_damage, snow_cover
- **building**: normal, crack
- **utility_pole**: normal, insulator_defect
- **machinery**: normal, crack, inclusion, patches, pitted_surface, rolled_in_scale, scratches, corrosion, anomaly

## Sources and licences

| source | Kaggle dataset | licence | images used | notes |
|---|---|---|---|---|
| pv_defect | [alicjalena/pv-panel-defect-dataset](https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset) | CC-BY-NC-SA-4.0 | 784 |  |
| solar_clean_faulty | [pythonafroz/solar-panel-images](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images) | unknown | 2 |  |
| pv_infrared | [marcosgabriel/infrared-solar-modules](https://www.kaggle.com/datasets/marcosgabriel/infrared-solar-modules) | not listed (check the Kaggle page) | 0 | disabled in config.yaml |
| crack_concrete | [arunrk7/surface-crack-detection](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) | copyright-authors | 1000 |  |
| sdnet2018 | [aniruddhsharma/structural-defects-network-concrete-crack-images](https://www.kaggle.com/datasets/aniruddhsharma/structural-defects-network-concrete-crack-images) | other | 1000 |  |
| uav_crack | [zara2099/uav-based-crack-detection-image-dataset](https://www.kaggle.com/datasets/zara2099/uav-based-crack-detection-image-dataset) | CC0-1.0 | 1000 | segmentation masks in sibling masks/: tiles without crack pixels → normal, tiles above the crack fraction → crack; masks consumed by mask_tiles |
| cplid_insulator | [mingyeiu/insulator-chinese-power-line-insulator-dataset](https://www.kaggle.com/datasets/mingyeiu/insulator-chinese-power-line-insulator-dataset) | MIT | 997 | VOC boxes: each insulator box is cropped; a box containing a 'defect' box centre → insulator_defect (defective images are synthetic composites per the dataset README) |
| powerline_parts | [abdulbasit89/power-line-components-images-dataset](https://www.kaggle.com/datasets/abdulbasit89/power-line-components-images-dataset) | CC0-1.0 | 2003 | dropout cut-out (porcelain insulator body); upstream folder is misspelt 'defeftive'; transformer bushing is an insulator component |
| epri_distribution | [dexterlewis/epri-distribution-inspection-imagery](https://www.kaggle.com/datasets/dexterlewis/epri-distribution-inspection-imagery) | not listed (check the Kaggle page) | 0 |  |
| neu_steel | [kaustubhdikshit/neu-surface-defect-database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) | unknown | 1799 | crazing = network of fine surface cracks |
| corrosion | [shreyaspoharkar/corrosion-and-noncorrosion-dataset](https://www.kaggle.com/datasets/shreyaspoharkar/corrosion-and-noncorrosion-dataset) | unknown | 0 | disabled in config.yaml |
| corrosion_detect | [wednesday233/corrosion-detect-dataset](https://www.kaggle.com/datasets/wednesday233/corrosion-detect-dataset) | MIT | 1217 | YOLO boxes in sibling labels/; images with empty label files are skipped (unlabelled, not verified clean) |
| corrosion_rustguard | [ananyagogi/corrosion-metal-rustguard](https://www.kaggle.com/datasets/ananyagogi/corrosion-metal-rustguard) | MIT | 555 | Roboflow polygons → bounding boxes; classes Metal / Metal corroido / Tornillo sin corrosion |
| mvtec_ad | [ipythonx/mvtec-ad](https://www.kaggle.com/datasets/ipythonx/mvtec-ad) | copyright-authors | 1184 | all other MVTec defect folders; defect_type is the folder name |

Licences are read from Kaggle metadata at download time. Datasets marked
`copyright-authors` or `unknown` are used here for research/evaluation only; check the
dataset page before any redistribution or commercial use.
