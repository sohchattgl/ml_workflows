import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
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
print(f'Train: {X_train.shape}, positives={y_train.sum()} ({y_train.mean():.2%})')
print(f'Val:   {X_val.shape}, positives={y_val.sum()} ({y_val.mean():.2%})')

# scale_pos_weight = neg / pos to up-weight the minority class
n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = n_neg / n_pos
print(f'scale_pos_weight = {n_neg}/{n_pos} = {spw:.2f}')


def evaluate(name, model, X, y):
    """Report AUROC, PR-AUC, F1 at default + optimal thresholds, confusion matrix."""
    proba = model.predict_proba(X)[:, 1]
    pred_default = (proba >= 0.5).astype(int)

    auroc = roc_auc_score(y, proba)
    ap = average_precision_score(y, proba)
    f1_default = f1_score(y, pred_default, zero_division=0)

    # Best F1 across all thresholds
    prec, rec, thr = precision_recall_curve(y, proba)
    f1_curve = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    best_idx = int(np.nanargmax(f1_curve[:-1]))  # last entry has no threshold
    f1_best = f1_curve[best_idx]
    best_thr = thr[best_idx]
    best_p, best_r = prec[best_idx], rec[best_idx]

    print(f'\n========== {name} — VAL metrics ==========')
    print(f'  ROC-AUC       : {auroc:.4f}')
    print(f'  PR-AUC        : {ap:.4f}  (random baseline = {y.mean():.4f})')
    print(f'  F1 @ thr=0.50 : {f1_default:.4f}')
    print(f'  F1 best       : {f1_best:.4f}  @ thr={best_thr:.4f}  (P={best_p:.3f}, R={best_r:.3f})')

    print('\n  Confusion matrix @ threshold=0.5:')
    cm = confusion_matrix(y, pred_default)
    print(f'    TN={cm[0,0]:>4}  FP={cm[0,1]:>4}')
    print(f'    FN={cm[1,0]:>4}  TP={cm[1,1]:>4}')

    print('\n  Confusion matrix @ best-F1 threshold:')
    pred_best = (proba >= best_thr).astype(int)
    cmb = confusion_matrix(y, pred_best)
    print(f'    TN={cmb[0,0]:>4}  FP={cmb[0,1]:>4}')
    print(f'    FN={cmb[1,0]:>4}  TP={cmb[1,1]:>4}')

    print('\n  Recall / Precision / F1 / Threshold trade-offs:')
    for target_recall in [0.5, 0.6, 0.7, 0.8]:
        ok = rec[:-1] >= target_recall
        if ok.any():
            idx = np.where(ok)[0][-1]
            f1_at = 2 * prec[idx] * rec[idx] / max(prec[idx] + rec[idx], 1e-9)
            print(f'    recall>={target_recall:.1f}: P={prec[idx]:.3f} R={rec[idx]:.3f} '
                  f'F1={f1_at:.3f} thr={thr[idx]:.4f}')
        else:
            print(f'    recall>={target_recall:.1f}: not achievable')

    return {
        'roc_auc': auroc, 'pr_auc': ap,
        'f1_default': f1_default, 'f1_best': f1_best, 'best_thr': best_thr,
    }


# ---------- Logistic Regression ----------
logreg = LogisticRegression(
    class_weight='balanced',
    max_iter=2000,
    solver='lbfgs',
    random_state=SEED,
)
logreg.fit(X_train, y_train)
lr_metrics = evaluate('Logistic Regression', logreg, X_val, y_val)

# ---------- XGBoost ----------
xgb = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=spw,
    eval_metric='aucpr',
    tree_method='hist',
    random_state=SEED,
    n_jobs=-1,
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_metrics = evaluate('XGBoost', xgb, X_val, y_val)

# ---------- Comparison ----------
print('\n========== SUMMARY (Val) ==========')
summary = pd.DataFrame({
    'Model':         ['Logistic Regression', 'XGBoost'],
    'ROC-AUC':       [lr_metrics['roc_auc'], xgb_metrics['roc_auc']],
    'PR-AUC':        [lr_metrics['pr_auc'], xgb_metrics['pr_auc']],
    'F1 @ 0.5':      [lr_metrics['f1_default'], xgb_metrics['f1_default']],
    'F1 best':       [lr_metrics['f1_best'], xgb_metrics['f1_best']],
    'Best thr':      [lr_metrics['best_thr'], xgb_metrics['best_thr']],
}).round(4)
print(summary.to_string(index=False))

# Save artifacts
joblib.dump(logreg, 'model_logreg.joblib')
joblib.dump(xgb, 'model_xgb.joblib')
print('\nSaved -> model_logreg.joblib, model_xgb.joblib')
