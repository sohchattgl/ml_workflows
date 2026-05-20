import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

TARGET = 'target_delinquent'
SEED = 42
N_ITER = 60       # random combinations to try
N_FOLDS = 5

train = pd.read_csv('data_train.csv')
val = pd.read_csv('data_val.csv')
X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
X_val, y_val = val.drop(columns=[TARGET]), val[TARGET]

n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = n_neg / n_pos
print(f'Train: {X_train.shape}, positives={n_pos} ({y_train.mean():.2%})')
print(f'scale_pos_weight (fixed): {spw:.2f}')

# ---------- Search space ----------
# Wide enough to explore, narrow enough to be tractable on this dataset
param_dist = {
    'n_estimators':     randint(100, 600),
    'max_depth':        randint(3, 9),
    'learning_rate':    loguniform(0.01, 0.2),
    'min_child_weight': randint(1, 15),
    'subsample':        uniform(0.6, 0.4),    # 0.6 - 1.0
    'colsample_bytree': uniform(0.6, 0.4),    # 0.6 - 1.0
    'gamma':            uniform(0.0, 1.0),
    'reg_alpha':        loguniform(1e-3, 1.0),
    'reg_lambda':       loguniform(1e-2, 5.0),
}

base = XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',
    tree_method='hist',
    scale_pos_weight=spw,
    random_state=SEED,
    n_jobs=1,        # parallelism is at the CV level
)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
search = RandomizedSearchCV(
    estimator=base,
    param_distributions=param_dist,
    n_iter=N_ITER,
    scoring='average_precision',   # PR-AUC
    cv=cv,
    n_jobs=-1,
    random_state=SEED,
    refit=True,                    # refit best on full train
    return_train_score=False,
    verbose=1,
)

t0 = time.time()
search.fit(X_train, y_train)
elapsed = time.time() - t0
print(f'\nSearch finished in {elapsed:.1f}s')

print(f'\nBest CV PR-AUC: {search.best_score_:.4f}')
print('Best params:')
for k, v in sorted(search.best_params_.items()):
    print(f'  {k}: {v}')

# CV score distribution
results = pd.DataFrame(search.cv_results_)
top5 = results.nlargest(5, 'mean_test_score')[['mean_test_score', 'std_test_score', 'params']]
print('\nTop 5 CV configurations:')
for _, row in top5.iterrows():
    print(f"  PR-AUC={row['mean_test_score']:.4f} ±{row['std_test_score']:.4f}")

# ---------- Evaluate tuned model on VAL ----------
best = search.best_estimator_
proba_val = best.predict_proba(X_val)[:, 1]
val_auroc = roc_auc_score(y_val, proba_val)
val_ap = average_precision_score(y_val, proba_val)
prec, rec, thr = precision_recall_curve(y_val, proba_val)
f1_curve = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
best_f1_idx = int(np.nanargmax(f1_curve[:-1]))
val_f1_best = f1_curve[best_f1_idx]
val_thr = thr[best_f1_idx]

print('\n========== Tuned XGBoost on VAL ==========')
print(f'  ROC-AUC       : {val_auroc:.4f}')
print(f'  PR-AUC        : {val_ap:.4f}  (random = {y_val.mean():.4f})')
print(f'  F1 best       : {val_f1_best:.4f}  @ thr={val_thr:.4f}')

pred_best = (proba_val >= val_thr).astype(int)
cm = confusion_matrix(y_val, pred_best)
print(f'\n  Confusion @ best-F1 threshold:')
print(f'    TN={cm[0,0]:>4}  FP={cm[0,1]:>4}')
print(f'    FN={cm[1,0]:>4}  TP={cm[1,1]:>4}')

# Baseline XGB metrics for direct comparison
baseline_metrics = {
    'roc_auc':   0.6493,
    'pr_auc':    0.0347,
    'f1_best':   0.0800,
}
print('\n========== Baseline vs Tuned (VAL) ==========')
comp = pd.DataFrame({
    'Metric':    ['ROC-AUC', 'PR-AUC', 'F1 best'],
    'Baseline':  [baseline_metrics['roc_auc'], baseline_metrics['pr_auc'], baseline_metrics['f1_best']],
    'Tuned':     [val_auroc, val_ap, val_f1_best],
}).round(4)
comp['Delta'] = (comp['Tuned'] - comp['Baseline']).round(4)
print(comp.to_string(index=False))

# Persist
joblib.dump(best, 'model_xgb_tuned.joblib')
with open('xgb_best_params.json', 'w') as f:
    json.dump(search.best_params_, f, indent=2, default=str)
print('\nSaved -> model_xgb_tuned.joblib, xgb_best_params.json')
