PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
export PYTHONPATH := $(CURDIR)

.PHONY: setup data train eval memory cloud api web test bench camera-probe smoke seed reid-capture reid-eval \
	lint sim-eval sim-quick train-reid train-change copy-paste train-detector train-seg train-thermal calibrate export-edge \
	owner-report retrain-feedback deploy-config

setup:
	test -d .venv || python3 -m venv .venv
	$(PIP) install -r requirements.txt
	cd frontend && npm ci

data:
	$(PY) scripts/download_datasets.py
	$(PY) scripts/prepare_unified_dataset.py

train:
	$(PY) training/train_condition_classifier.py

eval:
	$(PY) training/evaluate.py

memory:
	$(PY) scripts/migrate_json_store.py
	$(PY) scripts/seed_users.py
	$(PY) scripts/build_semantic_memory.py

cloud:
	$(PY) -m cloud.main

api:
	$(PY) run_api.py

web:
	cd frontend && npm run dev

test:
	$(PY) -m pytest -q

bench:
	$(PY) scripts/bench.py --assets 5000

camera-probe:
	$(PY) scripts/camera_probe.py

seed:
	$(PY) scripts/seed_users.py

smoke:
	$(PY) scripts/smoke_e2e.py

# usage: make reid-capture OBJECT=pump_1 SESSION=s1 CONDITION=normal
reid-capture:
	$(PY) scripts/capture_reid_set.py --object $(OBJECT) --session $(SESSION) --condition $(CONDITION)

reid-eval:
	$(PY) training/evaluate_reid.py

# ── upgrades ─────────────────────────────────────────────────────────────
lint:
	$(PY) -m ruff check .

sim-eval:            # 4.3 / 7.2 ablations → reports/sim/ablation.md
	$(PY) scripts/run_sim_eval.py

sim-quick:           # the CI gates
	$(PY) scripts/run_sim_eval.py --quick

train-reid:          # 1.3
	$(PY) training/train_reid_head.py

train-change:        # 2.1
	$(PY) training/train_change_detector.py

copy-paste:          # 3.5 synthetic physical_damage, then `make train`
	$(PY) training/copy_paste_augment.py

train-detector:      # 3.2
	$(PY) training/train_detector.py

train-seg:           # 3.3
	$(PY) training/train_defect_seg.py

train-thermal:       # 3.1
	$(PY) training/train_thermal.py

calibrate:           # 3.6 usage: make calibrate IMAGES='calib/*.jpg'
	$(PY) scripts/calibrate_camera.py --images $(IMAGES)

export-edge:         # 4.1
	$(PY) scripts/export_edge.py --out models/edge

owner-report:        # 6.4 usage: make owner-report SITE=site_demo MONTH=2026-10
	$(PY) scripts/owner_report.py --site $(SITE) --month $(MONTH)

retrain-feedback:    # 5.4
	$(PY) training/retrain_from_feedback.py

deploy-config:       # 6.1 usage: make deploy-config DOMAIN=audi.example.org
	$(PY) scripts/make_cloud_config.py $(DOMAIN)
