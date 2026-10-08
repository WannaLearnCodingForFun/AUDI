# AUDI — AI-Powered Persistent Semantic Memory for Intelligent Drone Inspection

AUDI provides the ML component for **Persistent Semantic Memory**: asset re-identification, visual similarity matching, and historical condition comparison across multiple drone inspection missions.

## Installation

```bash
cd AUDI
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch, torchvision, NumPy, Pillow, PyYAML.

DINOv2 is loaded via `torch.hub` on first run (requires internet). If unavailable, the system automatically falls back to torchvision ResNet50, then to a spatial-feature projection embedder for fully offline operation.

**For best matching accuracy**, install pretrained weights:
```bash
# macOS Python SSL fix (if weight download fails)
/Applications/Python\ 3.*/Install\ Certificates.command
pip install -r requirements.txt
```

The demo uses `config_demo.yaml` (stricter `similarity_threshold: 0.98`) when running offline with the projection embedder. Production deployments with DINOv2/ResNet50 should use `config.yaml` (threshold 0.75).

## Quick Start (Full System with Kaggle Dataset + Frontend)

```bash
cd AUDI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Download Kaggle PV Panel Defect Dataset
kaggle datasets download -d alicjalena/pv-panel-defect-dataset -p dataset/raw --unzip

# 2. Prepare balanced subset (~360 images)
PYTHONPATH=. python scripts/prepare_pv_dataset.py

# 3. Build semantic memory from dataset
PYTHONPATH=. python scripts/build_semantic_memory.py

# 4. Start backend API (terminal 1)
PYTHONPATH=. python run_api.py

# 5. Start frontend (terminal 2)
cd frontend && npm install && npm run dev
# Open http://localhost:5173
```

## Quick Start (Original Synthetic Demo)

```bash
# 1. Generate synthetic demo dataset
python scripts/generate_demo_dataset.py

# 2. Run multi-mission demo
python demo.py

# 3. Visualize memory state
python visualize_memory.py
python visualize_memory.py --plot

# 4. Run tests
pytest tests/ -v

# 5. Evaluate ML components
python training/evaluate_model.py
```

## Project Structure

```
AUDI/
├── ml/                     # Core ML pipeline (embedding, matching, change detection)
├── memory/                 # Vector search + local/cloud storage abstraction
├── api/                    # FastAPI backend
├── frontend/               # React + Vite dashboard
├── training/               # Optional fine-tuning and evaluation
├── scripts/                # Dataset preparation utilities
├── dataset/                # Kaggle PV subset + synthetic demo data
├── tests/                  # Unit and integration tests
├── demo.py                 # Multi-mission persistent memory demo
├── run_api.py              # Start FastAPI backend
├── visualize_memory.py     # Memory state visualization
└── config.yaml             # All thresholds and weights
```

## ML Architecture

```
Live Camera Frame
       ↓
Asset Detection (external — bbox crop supported)
       ↓
Asset Crop → AssetEncoder
       ↓
Visual Embedding Model (DINOv2 / ResNet50 fallback)
       ↓
Normalized Embedding Vector
       ↓
Vector Similarity Search (NumPy dev / FAISS-ready interface)
       ↓
Persistent Semantic Memory Store
       ↓
┌─────────────────────┐
│ Existing Asset      │ → retrieve history → ChangeDetector → Severity → Update
│ OR New Asset        │ → register baseline → store embedding
└─────────────────────┘
```

### Key Design Principles

1. **Asset Identity ≠ Condition Change** — matching (same physical asset?) and change detection (has condition changed?) are separate operations.
2. **Database Decoupling** — `SemanticMemoryStore` abstraction; local JSON store works offline; cloud interface ready for future sync.
3. **Model Replaceability** — embedding model, vector backend, and change detector are swappable without redesigning the pipeline.
4. **No Magic Numbers** — all thresholds in `config.yaml`.

## Where the Embedding Model Is Used

| Stage | Usage |
|-------|-------|
| New asset registration | Generate initial embedding → store in vector memory |
| Existing asset match | Query embedding vs. all stored embeddings (cosine similarity) |
| Change detection | Compare current embedding vs. historical embedding sequence |
| Optional fine-tuning | `training/train_embedding_model.py` with real labeled data |

## How Asset Matching Works

1. Encode cropped asset image → L2-normalized embedding vector.
2. Search vector memory for top-k candidates.
3. Compute cosine similarity against all candidates.
4. If best similarity ≥ `matching.similarity_threshold` (default 0.75) → **existing asset**.
5. Otherwise → **register new asset**.

```python
from ml.semantic_memory import SemanticMemoryEngine

engine = SemanticMemoryEngine()
result = engine.process_observation(image, asset_type="solar_panel")
# result: asset_id, is_new_asset, match_confidence, change_magnitude, severity_score
```

## How Change Magnitude Is Calculated

`ChangeDetector.compare()` computes:

```
combined_similarity = 0.6 × max(historical_similarities) + 0.4 × mean(historical_similarities)
change_magnitude = clip(1 - combined_similarity, 0, 1)
```

- **0** = no meaningful change (current observation highly similar to history)
- **1** = very large change (low similarity to all historical observations)

This is intentionally separate from asset identity matching — the same asset can have high identity similarity but significant condition change when damage appears.

## Severity and Revisit Priority

**Severity** (SRS formula):
```
S = (w1 × M + w2 × C) × Cf
```
- M = change magnitude (from ML)
- C = asset criticality (from config, per asset type)
- Cf = observation/match confidence (from ML)

**Revisit Priority**:
```
R = w3 × Sr + w4 × T + w5 × D
```
- Sr = recent severity
- T = historical severity trend rate
- D = normalized time since last inspection

Default weights and thresholds are in `config.yaml`.

## Persistent Memory Across Missions

AUDI does **not** treat each flight independently:

| Mission | Behavior |
|---------|----------|
| Mission 1 | Solar Panel A → no match → **new asset** → baseline stored |
| Mission 2 | Solar Panel A → embedding search → **matched** → history retrieved → observation appended |
| Mission 3 | Solar Panel A (damaged) → **matched** → change detected → severity calculated → revisit priority updated |
| Mission 3 | Building B → no match → **new asset** registered |

Run `python demo.py` to see this scenario with printed memory state after each mission.

Memory persists to `data/demo_semantic_memory.json` (configurable via `config.yaml`).

## Dataset Setup (Kaggle PV Panel Defect Dataset)

**Source:** [PV Panel Defect Dataset](https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset)

Six classes: Clean, Dusty, Bird-drop, Electrical-damage, Physical-damage, Snow-covered.

```bash
# Requires Kaggle CLI credentials (~/.kaggle/kaggle.json)
kaggle datasets download -d alicjalena/pv-panel-defect-dataset -p dataset/raw --unzip
PYTHONPATH=. python scripts/prepare_pv_dataset.py
PYTHONPATH=. python scripts/build_semantic_memory.py
```

See `dataset/README.md` for class mappings, subset rationale, and **Simulated Multi-Mission History** limitations.

## Running Backend

```bash
cd AUDI
source .venv/bin/activate
PYTHONPATH=. python run_api.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

**API Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/summary` | Dashboard summary |
| GET | `/api/assets` | List all assets |
| GET | `/api/assets/{id}` | Asset details |
| GET | `/api/assets/{id}/history` | Observation history + severity trend |
| GET | `/api/memory/search?q=` | Search semantic memory |
| POST | `/api/analyze` | Upload image for analysis |
| POST | `/api/simulate-mission` | Run simulated inspection mission |

## Running Frontend

```bash
cd AUDI/frontend
npm install
npm run dev
# Open http://localhost:5173
```

The dashboard includes: **Dashboard**, **Assets**, **Live Analysis**, **Mission History**, **Semantic Memory**.

All values are fetched from the backend API — no hardcoded data.

## Demo — Simulated Multi-Mission Inspection

1. Open http://localhost:5173
2. **Dashboard** shows 356 assets, defect counts, priority list
3. **Assets** → click an asset with simulated history (e.g. search `asset_efc116ea`)
4. View **Observation Timeline** with severity trend chart
5. **Live Analysis** → upload a PV panel image → see match/change/severity
6. **Mission History** → click **Simulate Mission** → processes 8 images, updates memory
7. Refresh page — assets and history persist in `data/pv_semantic_memory.json`

**Research Demonstration:** Kaggle images provide realistic defect examples. Multi-mission histories are **simulated** — not real longitudinal drone data.

## Demo vs Real Data

| | Demo/Synthetic | Real Training Data |
|---|---|---|
| Source | `generate_demo_dataset.py` | Drone inspection images |
| Purpose | Pipeline testing | Production fine-tuning |
| Accuracy | **Not production accurate** | Requires labeled dataset |

## Configuration

Edit `config.yaml` for all parameters:

```yaml
matching:
  similarity_threshold: 0.75
severity:
  w1: 0.6
  w2: 0.4
  priority_threshold: 0.5
runtime:
  frame_skip: 5
```

## Optional Fine-Tuning

Requires **real** labeled data (not synthetic demo):

```bash
python training/train_embedding_model.py \
  --data dataset \
  --epochs 10 \
  --batch-size 16 \
  --lr 1e-4
```

The pretrained model works without fine-tuning.

## Real-Time Considerations

- Models loaded once at engine initialization
- Configurable frame skipping (`runtime.frame_skip`)
- GPU auto-detection (CUDA if available)
- Timing metrics exposed: `embedding_latency_ms`, `matching_latency_ms`, `total_processing_latency_ms`
- Errors logged without crashing the inspection pipeline

## Testing

```bash
pytest tests/ -v
```

Tests cover: embedding normalization, similarity, matching thresholds, memory persistence, change detection bounds, severity/revisit formulas, and end-to-end multi-observation flow.

## License

Part of the AUDI drone inspection system. No autonomous drone control is implemented in this module.
