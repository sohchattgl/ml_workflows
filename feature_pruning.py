import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from xgboost import XGBClassifier

TARGET = 'target_delinquent'
SEED = 42

train = pd.read_csv('data_train.csv')
val = pd.read_csv('data_val.csv')
X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
X_val, y_val = val.drop(columns=[TARGET]), val[TARGET]

tuned = joblib.load('model_xgb_tuned.joblib')
with open('xgb_best_params.json') as f:
    raw = json.load(f)

# Cast hyperparams to native numeric types
best_params = {}
for k, v in raw.items():
    try:
        best_params[k] = int(v) if float(v).is_integer() and k in {
            'n_estimators', 'max_depth', 'min_child_weight'
        } else float(v)
    except (ValueError, TypeError):
        best_params[k] = v

# ---------- Gain-based importance ----------
booster = tuned.get_booster()
gain = booster.get_score(importance_type='gain')
imp = pd.DataFrame({
    'feature': X_train.columns,
    'gain': [gain.get(f, 0.0) for f in X_train.columns],
}).sort_values('gain', ascending=False).reset_index(drop=True)
imp['cum_pct'] = imp['gain'].cumsum() / imp['gain'].sum() * 100

n_zero = (imp['gain'] == 0).sum()
print(f'Total features: {len(imp)}  |  zero-gain: {n_zero}')
print('\nTop 15 by gain:')
print(imp.head(15).round(4).to_string(index=False))
print('\nBottom 5 non-zero:')
print(imp[imp.gain > 0].tail(5).round(4).to_string(index=False))

# ---------- Pruning ----------
# Keep features that contributed any gain; drop zero-gain ones
keep = imp[imp.gain > 0]['feature'].tolist()
drop = imp[imp.gain == 0]['feature'].tolist()
print(f'\nKeeping {len(keep)} features, dropping {len(drop)}')
print('Dropped:', drop[:10], '...' if len(drop) > 10 else '')

# ---------- Refit on pruned features ----------
n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = n_neg / n_pos
xgb_pruned = XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',
    tree_method='hist',
    scale_pos_weight=spw,
    random_state=SEED,
    n_jobs=-1,
    **best_params,
)
xgb_pruned.fit(X_train[keep], y_train)

# ---------- Evaluate ----------
def report(label, model, X, y):
    proba = model.predict_proba(X)[:, 1]
    auroc = roc_auc_score(y, proba)
    ap = average_precision_score(y, proba)
    prec, rec, thr = precision_recall_curve(y, proba)
    f1c = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    bi = int(np.nanargmax(f1c[:-1]))
    t = thr[bi]
    pred = (proba >= t).astype(int)
    cm = confusion_matrix(y, pred)
    return {
        'Model': label, 'ROC-AUC': auroc, 'PR-AUC': ap, 'F1 best': f1c[bi],
        'Thr': t, 'TN': cm[0,0], 'FP': cm[0,1], 'FN': cm[1,0], 'TP': cm[1,1],
    }

rows = [
    report('Tuned (all 73 feats)', tuned, X_val, y_val),
    report(f'Pruned ({len(keep)} feats)', xgb_pruned, X_val[keep], y_val),
]
df = pd.DataFrame(rows)
for col in ['ROC-AUC', 'PR-AUC', 'F1 best', 'Thr']:
    df[col] = df[col].round(4)
print('\n========== Pruned vs Tuned (VAL) ==========')
print(df.to_string(index=False))

joblib.dump({'model': xgb_pruned, 'features': keep}, 'model_xgb_pruned.joblib')
print('\nSaved -> model_xgb_pruned.joblib')
