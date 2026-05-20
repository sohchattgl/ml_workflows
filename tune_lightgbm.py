import json
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

TARGET = 'target_delinquent'
SEED = 42
N_ITER = 60
N_FOLDS = 5

train = pd.read_csv('data_train.csv')
val = pd.read_csv('data_val.csv')
X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
X_val, y_val = val.drop(columns=[TARGET]), val[TARGET]

n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = n_neg / n_pos
print(f'Train: {X_train.shape}, positives={n_pos} ({y_train.mean():.2%})')
print(f'scale_pos_weight: {spw:.2f}')

# Sanitize feature names (LightGBM dislikes some chars)
X_train.columns = [c.replace(' ', '_') for c in X_train.columns]
X_val.columns = X_train.columns

param_dist = {
    'n_estimators':     randint(100, 600),
    'max_depth':        randint(3, 9),
    'num_leaves':       randint(8, 128),
    'learning_rate':    loguniform(0.01, 0.2),
    'min_child_samples': randint(5, 60),
    'subsample':        uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_alpha':        loguniform(1e-3, 1.0),
    'reg_lambda':       loguniform(1e-2, 5.0),
}

base = LGBMClassifier(
    objective='binary',
    scale_pos_weight=spw,
    random_state=SEED,
    n_jobs=1,
    verbosity=-1,
)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
search = RandomizedSearchCV(
    estimator=base,
    param_distributions=param_dist,
    n_iter=N_ITER,
    scoring='average_precision',
    cv=cv,
    n_jobs=-1,
    random_state=SEED,
    refit=True,
    verbose=1,
)

t0 = time.time()
search.fit(X_train, y_train)
print(f'\nLightGBM search finished in {time.time()-t0:.1f}s')
print(f'Best CV PR-AUC: {search.best_score_:.4f}')
print('Best params:')
for k, v in sorted(search.best_params_.items()):
    print(f'  {k}: {v}')

best = search.best_estimator_

# Evaluate on val
proba = best.predict_proba(X_val)[:, 1]
auroc = roc_auc_score(y_val, proba)
ap = average_precision_score(y_val, proba)
prec, rec, thr = precision_recall_curve(y_val, proba)
f1c = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
bi = int(np.nanargmax(f1c[:-1]))
t_best = thr[bi]
f1_best = f1c[bi]
pred = (proba >= t_best).astype(int)
cm = confusion_matrix(y_val, pred)

print('\n========== LightGBM tuned on VAL ==========')
print(f'  ROC-AUC : {auroc:.4f}')
print(f'  PR-AUC  : {ap:.4f}  (random = {y_val.mean():.4f})')
print(f'  F1 best : {f1_best:.4f}  @ thr={t_best:.4f}')
print(f'  Confusion @ best-F1:')
print(f'    TN={cm[0,0]:>4}  FP={cm[0,1]:>4}')
print(f'    FN={cm[1,0]:>4}  TP={cm[1,1]:>4}')

# Compare with tuned XGBoost (from earlier run)
xgb_val = {'ROC-AUC': 0.6865, 'PR-AUC': 0.0506, 'F1 best': 0.1429, 'TP': 7, 'FP': 73}
print('\n========== XGBoost vs LightGBM (VAL) ==========')
comp = pd.DataFrame([
    {'Model': 'XGBoost (tuned)', **xgb_val},
    {'Model': 'LightGBM (tuned)', 'ROC-AUC': auroc, 'PR-AUC': ap,
     'F1 best': f1_best, 'TP': int(cm[1,1]), 'FP': int(cm[0,1])},
])
for col in ['ROC-AUC', 'PR-AUC', 'F1 best']:
    comp[col] = comp[col].round(4)
print(comp.to_string(index=False))

joblib.dump(best, 'model_lgbm_tuned.joblib')
with open('lgbm_best_params.json', 'w') as f:
    json.dump(search.best_params_, f, indent=2, default=str)
print('\nSaved -> model_lgbm_tuned.joblib, lgbm_best_params.json')
