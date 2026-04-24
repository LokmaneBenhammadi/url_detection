# url_detection_project

AI-based malicious URL detection. Given a URL, the system predicts whether
it is **benign** or **malicious** (phishing / malware / defacement / spam)
using classical Machine Learning and Deep Learning models trained on
pre-extracted lexical features from the ISCX-URL2016 dataset.

## 1) Architecture Overview

```text
Raw URL
  -> Feature Extraction (78 numeric lexical features, ISCX extractor)
  -> Preprocessing        (sentinel -1 -> NaN, median imputation, StandardScaler)
  -> Feature Selection    (Mutual Information, top-12)
  -> Classification       (LR / NB / KNN / DT / RF / XGBoost / AdaBoost / SVM / MLP / DNN)
  -> Hyperparameter Tuning (GridSearch on RF, PSO on XGBoost)
  -> Evaluation           (Accuracy ↑, FPR ↓ as primary criteria)
  -> Report / JSON
```

Binary target: `benign -> 0`, any other class -> 1. The binary collapse
is dictated by the **False Positive Rate (FAR)** metric required by the
course rubric, which is only well-defined in a two-class setting.

## 2) Project Structure

```text
url_detection_project/
├── data/
│   ├── All_clean.csv                   # 26,953 URLs x 78 numeric features
│   ├── All_BestFirst.csv               # Weka BestFirst reference subset (8 features)
│   ├── All_Infogain.csv                # Weka InfoGain reference subset (12 features)
│   └── {Defacement,Malware,Phishing,Spam}_*.csv   # per-class variants
├── notebooks/
│   ├── 01_EDA.ipynb                              # exploratory data analysis
│   ├── 02_Feature_Selection_Comparison.ipynb     # MI / RF-entropy / RFE / SFS vs refs
│   ├── 03_Model_Training_and_Evaluation.ipynb    # ML + DL + tuning + report
│   └── outputs/
│       ├── eda/
│       ├── feature_selection_comparison/
│       └── model_training/                       # CSVs, PNGs, conclusion.md
├── src/
│   ├── __init__.py
│   ├── dataset.py                      # (placeholder) CSV loading + splits
│   ├── preprocess.py                   # (placeholder) imputation + scaling
│   ├── features.py                     # (placeholder) 12-feature selector
│   ├── model.py                        # (placeholder) sklearn / keras model factory
│   ├── train.py                        # (placeholder) CLI training entrypoint
│   ├── evaluate.py                     # (placeholder) metrics incl. FPR/FAR
│   └── inference.py                    # (placeholder) single-URL prediction
├── api/
│   ├── __init__.py
│   ├── main.py                         # (placeholder) FastAPI app
│   ├── schemas.py                      # (placeholder) pydantic request/response
│   └── utils.py                        # (placeholder) URL -> feature-vector glue
├── configs/
│   └── config.yaml                     # (placeholder) paths, seed, hyperparameters
├── checkpoints/
│   └── .gitkeep
├── Dockerfile                          # (placeholder)
├── requirements.txt
└── README.md
```

The project is currently **notebook-driven**; the `src/` and `api/`
trees are scaffolded for the upcoming productisation phase (see the
Roadmap).

## 3) Installation

### venv + pip (recommended)

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Core runtime dependencies: `numpy`, `pandas`, `scikit-learn`, `xgboost`,
`tensorflow`, `matplotlib`, `seaborn`, `nbformat`, `jupyter`.

### Conda (alternative)

```bash
conda create -n url_detection python=3.11 -y
conda activate url_detection
pip install -r requirements.txt
```

## 4) Docker (Containerization)

Build the image:

```bash
docker build -t url-detection:latest .
```

Run the containerized API:

```bash
docker run --rm -p 8000:8000 url-detection:latest
```

Test health endpoint:

```bash
curl -X GET http://127.0.0.1:8000/health
```

> Note: Docker + API are scaffolded but not yet implemented (see
> Roadmap). The notebooks run standalone.

## 5) Dataset Source

**ISCX-URL2016** — Canadian Institute for Cybersecurity, University of
New Brunswick.

- 26,953 URLs, 5 classes: `benign`, `phishing`, `malware`, `defacement`, `spam`
- 78 pre-extracted numeric lexical features (URL length, token counts,
  character-continuity rate, entropy, digit/letter ratios, etc.)
- Reference: https://www.unb.ca/cic/datasets/url-2016.html

Two reference feature subsets ship with the dataset and are used as
ground truth in notebook 02:

| File | Method | # features |
|---|---|---|
| `All_BestFirst.csv` | Weka BestFirst wrapper search | 8 |
| `All_Infogain.csv`  | Weka InfoGain Attribute Eval | 12 |

## 6) Pipeline — Notebooks

The three notebooks are designed to be executed in order.

### 6.1 — `01_EDA.ipynb`

Exploratory analysis of `All_clean.csv`: class balance, per-feature
distributions, sentinel-value audit (`-1` means "field absent" in the
ISCX extractor), correlation structure, outlier inspection.

### 6.2 — `02_Feature_Selection_Comparison.ipynb`

Compares four feature-selection methods, each *mathematically aligned*
with a Weka reference shipped in the dataset:

| Method | Aligns with |
|---|---|
| Mutual Information (`mutual_info_classif`) | `All_Infogain.csv` |
| Random Forest with `criterion='entropy'`   | `All_Infogain.csv` |
| Recursive Feature Elimination (RFE)        | `All_BestFirst.csv` |
| Sequential Forward Selection (SFS)         | `All_BestFirst.csv` |

Selection rule: **maximise `avg_jaccard` vs references, tie-break on
`avg_precision@K`**. Winner = **Mutual Information, K=12**, with 11/12
overlap against the Infogain reference.

Final 12 selected features:

```
Entropy_Domain, argDomanRatio, NumberRate_FileName, CharacterContinuityRate,
argPathRatio, ArgUrlRatio, pathurlRatio, domainUrlRatio, domainlength,
NumberRate_AfterPath, NumberofDotsinURL, domain_token_count
```

### 6.3 — `03_Model_Training_and_Evaluation.ipynb`

Trains and compares 11 models on the 12-feature subset.

| Family | Models |
|---|---|
| Classical ML | LogisticRegression, GaussianNB, KNN, DecisionTree, RandomForest, XGBoost, AdaBoost, SVM (RBF) |
| Deep Learning | MLP (sklearn), DNN (Keras, 128→64→32 + BN + Dropout), **Improved DNN** (L2 + He init + ReduceLROnPlateau + class weights) |
| Tuned | RF via `GridSearchCV`, XGBoost via custom **Particle Swarm Optimization** |

Evaluation metrics: `Accuracy`, `Precision`, `Recall`, `F1`, `ROC-AUC`,
`FPR` (= `FP / (FP + TN)`). Final ranking rule imposed by the course:
**Accuracy ↓, FPR ↑** (lexicographic tie-break).

## 7) How to Run

All three notebooks run end-to-end with:

```bash
source venv/bin/activate
jupyter lab notebooks/
```

Or execute headless:

```bash
jupyter nbconvert --to notebook --execute \
  notebooks/03_Model_Training_and_Evaluation.ipynb \
  --output 03_Model_Training_and_Evaluation.ipynb
```

Artifacts (CSVs, PNGs, `conclusion.md`) land under
`notebooks/outputs/<stage>/`.

## 8) How to Train (Placeholder — future `src/` CLI)

```bash
python -m src.train --config configs/config.yaml
```

## 9) How to Run Inference (Placeholder)

```bash
python -m src.inference --config configs/config.yaml --url "http://example.com/login?id=42"
```

## 10) How to Run the API (Placeholder)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### GET /health

```bash
curl -X GET http://127.0.0.1:8000/health
```

```json
{ "status": "ok" }
```

### POST /predict

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"url": "http://example.com/login?id=42"}'
```

Expected response shape (once implemented):

```json
{
  "url": "http://example.com/login?id=42",
  "prediction": "malicious",
  "probability": 0.87,
  "features_used": 12
}
```

## 11) Tech Stack

| Component | Technology |
|---|---|
| Classical ML        | scikit-learn, XGBoost |
| Deep Learning       | TensorFlow / Keras |
| Hyperparameter Tuning | GridSearchCV, custom PSO |
| Data / EDA          | pandas, numpy, matplotlib, seaborn |
| Notebooks           | Jupyter, nbformat |
| API Layer (planned) | FastAPI |
| Containerization (planned) | Docker |

## 12) Key Results

Driven by the course criterion **Accuracy ↑, FPR ↓**. The full ranking
is regenerated in
`notebooks/outputs/model_training/model_evaluation_results.csv` and
summarised in `conclusion.md`. Top models typically come from the
tree-ensemble family (RF / XGBoost) after tuning, with the Improved DNN
competitive on FPR thanks to class-weight balancing.