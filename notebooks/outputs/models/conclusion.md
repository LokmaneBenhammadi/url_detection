# Conclusion — Détection d'URLs malveillantes

## Critères officiels

> Accuracy ↑, FPR ↓

## Modèle gagnant

**XGBoost_PSO** (famille : `Tuned`)

| Métrique  | Valeur |
|-----------|--------|
| Accuracy  | 0.9848 |
| Precision | 0.9905 |
| Recall    | 0.9885 |
| F1-score  | 0.9895 |
| ROC-AUC   | 0.9982 |
| **FPR**   | **0.0248** |

## Second

`XGBoost` — Accuracy 0.9833, FPR 0.0221.

## Lecture

Parmi les 13 modèles comparés (ML classiques, Deep 
Learning, et versions optimisées par GridSearch + PSO), le modèle 
`XGBoost_PSO` domine selon le tri lexicographique imposé : 
il combine la plus haute Accuracy avec le plus faible taux de 
fausses alarmes parmi les ex-æquo en Accuracy.

## Fichiers générés

- `model_evaluation_results.csv`
- `bar_accuracy_f1.png`
- `bar_fpr.png`
- `roc_top3.png`
- `dnn_training_curves.png`