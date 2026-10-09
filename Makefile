PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
export PYTHONPATH := $(CURDIR)

.PHONY: setup data train eval memory cloud api web test bench camera-probe smoke seed reid-capture reid-eval

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
