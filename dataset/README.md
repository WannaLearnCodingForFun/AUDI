# PV Panel Defect Dataset (Kaggle Subset)

## Source

**Dataset:** [PV Panel Defect Dataset](https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset)  
**Author:** alicjalena (Kaggle)

## Classes (6)

| Kaggle Class | AUDI Condition |
|--------------|----------------|
| Clean | Normal |
| Dusty | Soiling |
| Bird-drop | Surface contamination |
| Electrical-damage | Electrical defect |
| Physical-damage / Physical-Damage | Physical defect |
| Snow-covered / Snow-Covered | Surface obstruction |

## Subset Used

Approximately **50–70 images per class** (~300–400 total), selected automatically from the downloaded Kaggle dataset using `scripts/prepare_pv_dataset.py`.

The full dataset contains ~1,500 images. We use a balanced subset for:

- Faster embedding generation during development
- Manageable storage for research/educational experimentation
- Sufficient class diversity for pipeline testing

## Directory Layout

```
dataset/
  raw/          # Full Kaggle download (NOT committed to Git)
  small/        # Balanced subset (NOT committed to Git)
    Bird-drop/
    Clean/
    Dusty/
    Electrical-damage/
    Physical-Damage/
    Snow-covered/
    metadata.csv
```

## Important Limitations

### Not a Longitudinal Drone Dataset

This Kaggle dataset provides **independent labeled images by defect class**. It does **not** contain repeated observations of the same physical solar panel across multiple drone missions.

### Simulated Multi-Mission Histories

AUDI demonstrates persistent semantic memory using **controlled simulated histories** built on top of these images:

- `DEMO_ASSET_A`: Clean → Dusty → Physical-damage
- `DEMO_ASSET_B`: Clean → Clean → Bird-drop
- `DEMO_ASSET_C`: Clean → Electrical-damage

These are labeled **`SIMULATED MULTI-MISSION HISTORY`** in metadata and the UI.

### Research / Educational Use

This integration is for **research demonstration** of the AUDI architecture and ML pipeline. It does **not** claim production-grade drone inspection accuracy.

## Setup

```bash
# Download full dataset from Kaggle
kaggle datasets download -d alicjalena/pv-panel-defect-dataset -p dataset/raw --unzip

# Create balanced subset
python scripts/prepare_pv_dataset.py

# Build semantic memory
python scripts/build_semantic_memory.py
```

## metadata.csv Columns

| Column | Description |
|--------|-------------|
| `image_id` | Unique image identifier |
| `image_path` | Path relative to `dataset/` |
| `class` | Original Kaggle class label |
| `split` | Always `small` for subset |
| `asset_id` | DEMO ASSET identifier (simulated) |
| `mission_id` | Simulated mission (e.g. `mission_001`) |
| `demo_asset` | `true` — not a real physical asset ID |
| `simulated_history` | `true` if part of multi-mission sequence |
| `note` | Limitation disclaimer |
