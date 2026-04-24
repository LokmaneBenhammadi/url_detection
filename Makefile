PYTHON  ?= python
PIP     ?= pip
JUPYTER ?= jupyter
UVICORN ?= uvicorn
HOST    ?= 0.0.0.0
PORT    ?= 8000
CONFIG  ?= configs/config.yaml
URL     ?= http://example.com/login?id=42

NOTEBOOK_DIR := notebooks
NB_EDA       := $(NOTEBOOK_DIR)/01_EDA.ipynb
NB_FS        := $(NOTEBOOK_DIR)/02_Feature_Selection_Comparison.ipynb
NB_TRAIN     := $(NOTEBOOK_DIR)/03_Model_Training_and_Evaluation.ipynb
OUTPUTS_DIR  := $(NOTEBOOK_DIR)/outputs

.PHONY: help install lab notebooks eda feature-selection train-nb \
        train infer run-api check clean clean-outputs

help:
	@echo "Available targets:"
	@echo "  make install            - Install dependencies from requirements.txt"
	@echo "  make lab                - Launch JupyterLab rooted at notebooks/"
	@echo "  make notebooks          - Execute all three notebooks in order (headless)"
	@echo "  make eda                - Execute notebook 01 (EDA)"
	@echo "  make feature-selection  - Execute notebook 02 (feature selection)"
	@echo "  make train-nb           - Execute notebook 03 (model training & eval)"
	@echo "  make train              - Placeholder CLI training (future src/train.py)"
	@echo "  make infer URL=<url>    - Placeholder CLI inference (future src/inference.py)"
	@echo "  make run-api            - Placeholder FastAPI server (future api/main.py)"
	@echo "  make check              - Syntax-check any Python modules under src/ and api/"
	@echo "  make clean              - Remove Python cache artifacts"
	@echo "  make clean-outputs      - Remove generated notebook outputs (CSVs, PNGs, MD)"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lab:
	$(JUPYTER) lab --notebook-dir=$(NOTEBOOK_DIR)

# ─── Notebook execution (headless, in-place) ────────────────────────────
notebooks: eda feature-selection train-nb

eda:
	$(JUPYTER) nbconvert --to notebook --execute --inplace $(NB_EDA)

feature-selection:
	$(JUPYTER) nbconvert --to notebook --execute --inplace $(NB_FS)

train-nb:
	$(JUPYTER) nbconvert --to notebook --execute --inplace $(NB_TRAIN)

# ─── Future src/ CLI entrypoints (scaffolded, not yet implemented) ──────
train:
	$(PYTHON) -m src.train --config $(CONFIG)

infer:
	$(PYTHON) -m src.inference --config $(CONFIG) --url "$(URL)"

run-api:
	$(UVICORN) api.main:app --host $(HOST) --port $(PORT) --reload

# ─── Maintenance ────────────────────────────────────────────────────────
check:
	@echo "Syntax-checking *.py under src/ and api/ (if any)..."
	@find src api -type f -name '*.py' 2>/dev/null | xargs -r $(PYTHON) -m py_compile

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +

clean-outputs:
	rm -rf $(OUTPUTS_DIR)/eda
	rm -rf $(OUTPUTS_DIR)/feature_selection_comparison
	rm -rf $(OUTPUTS_DIR)/feature_selection_v2
	rm -rf $(OUTPUTS_DIR)/model_training
