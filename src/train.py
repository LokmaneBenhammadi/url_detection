from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

from src.features import (
    SELECTED_FEATURES,
    build_feature_matrix,
    build_targets,
    binary_label_encoder,
    load_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = CHECKPOINT_DIR / 'best_model.joblib'
RANDOM_SEED = 42


def train() -> None:
    df_raw = load_dataset(DATA_DIR)
    print(f'Loaded dataset with shape: {df_raw.shape}')

    X_df = build_feature_matrix(df_raw).apply(pd.to_numeric, errors='coerce')
    y_binary, y_multiclass, label_encoder = build_targets(df_raw)

    n_sentinel = int((X_df == -1).sum().sum())
    X_df = X_df.replace(-1, np.nan)
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    n_nan = int(X_df.isna().sum().sum())
    print(f'Sentinel -1 values replaced: {n_sentinel:,}')
    print(f'Total NaN after replacement: {n_nan:,}')

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_train, X_test, y_train_binary, y_test_binary, y_train_multiclass, y_test_multiclass = train_test_split(
        X_scaled,
        y_binary,
        y_multiclass,
        test_size=0.20,
        stratify=y_binary,
        random_state=RANDOM_SEED,
    )

    print(f'X_train: {X_train.shape}, X_test: {X_test.shape}')
    print(f'Binary target distribution: benign={(y_binary == 0).sum()}, malicious={(y_binary == 1).sum()}')

    binary_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=len(y_binary) / (2 * sum(y_binary == 1)),  # for imbalanced
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1,
    )
    binary_model.fit(X_train, y_train_binary)

    n_classes = len(np.unique(y_multiclass))
    multiclass_model = XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1,
    )
    multiclass_model.fit(X_train, y_train_multiclass)

    print('\nBinary model evaluation on held-out test set:')
    y_pred_binary = binary_model.predict(X_test)
    print(classification_report(y_test_binary, y_pred_binary, digits=4))

    print('Multiclass model evaluation on held-out test set:')
    y_pred_multi = multiclass_model.predict(X_test)
    print(classification_report(y_test_multiclass, y_pred_multi, digits=4, target_names=label_encoder.classes_))

    bundle = {
        'models': {
            'binary': binary_model,
            'multiclass': multiclass_model,
        },
        'model_names': {
            'binary': 'XGBoost_binary',
            'multiclass': 'XGBoost_multiclass',
        },
        'imputer': imputer,
        'scaler': scaler,
        'selected_features': SELECTED_FEATURES,
        'label_encoder': label_encoder,
        'classes': label_encoder.classes_.tolist(),
    }

    joblib.dump(bundle, CHECKPOINT_PATH)
    print(f'Saved trained bundle to {CHECKPOINT_PATH}')


if __name__ == '__main__':
    train()
