import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

TARGET = 'target_delinquent'

train = pd.read_csv('data_train.csv')
val = pd.read_csv('data_val.csv')
test = pd.read_csv('data_test.csv')

X_val, y_val = val.drop(columns=[TARGET]), val[TARGET]
X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

# Load the EXACT saved tuned model
tuned = joblib.load('model_xgb_tuned.joblib')

def eval_at_threshold(model, X, y, thr_to_use=None):
    proba = model.predict_proba(X)[:, 1]
    auroc = roc_auc_score(y, proba)
    ap = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    prec, rec, thr = precision_recall_curve(y, proba)
    f1c = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    bi = int(np.nanargmax(f1c[:-1]))
    best_thr = thr[bi]
    used_thr = best_thr if thr_to_use is None else thr_to_use
    pred = (proba >= used_thr).astype(int)
    f1 = f1_score(y, pred, zero_division=0)
    cm = confusion_matrix(y, pred)
    return {
        'roc_auc': auroc, 'pr_auc': ap, 'brier': brier,
        'best_f1': f1c[bi], 'best_thr': best_thr,
        'f1_at_used_thr': f1, 'used_thr': used_thr,
        'TN': cm[0,0], 'FP': cm[0,1], 'FN': cm[1,0], 'TP': cm[1,1],
        'proba': proba,
    }

# Step 1: get val-tuned threshold (using best-F1 on val)
val_metrics = eval_at_threshold(tuned, X_val, y_val)
val_thr = val_metrics['best_thr']
print('========== Saved tuned XGBoost on VAL ==========')
print(f"  ROC-AUC : {val_metrics['roc_auc']:.4f}")
print(f"  PR-AUC  : {val_metrics['pr_auc']:.4f}  (random={y_val.mean():.4f})")
print(f"  Brier   : {val_metrics['brier']:.4f}")
print(f"  F1 best : {val_metrics['best_f1']:.4f} @ thr={val_thr:.4f}")
print(f"  TN={val_metrics['TN']}  FP={val_metrics['FP']}")
print(f"  FN={val_metrics['FN']}  TP={val_metrics['TP']}")

# Step 2: apply same threshold on TEST
print('\n========== Saved tuned XGBoost on TEST ==========')
print(f'(Using val-tuned threshold = {val_thr:.4f})\n')
test_metrics = eval_at_threshold(tuned, X_test, y_test, thr_to_use=val_thr)
print(f"  ROC-AUC : {test_metrics['roc_auc']:.4f}")
print(f"  PR-AUC  : {test_metrics['pr_auc']:.4f}  (random={y_test.mean():.4f})")
print(f"  Brier   : {test_metrics['brier']:.4f}")
print(f"  F1 @ val-thr ({val_thr:.4f}): {test_metrics['f1_at_used_thr']:.4f}")
print(f"  Confusion @ val-thr:")
print(f"    TN={test_metrics['TN']:>5}  FP={test_metrics['FP']:>4}")
print(f"    FN={test_metrics['FN']:>5}  TP={test_metrics['TP']:>4}")
print()
# For curiosity only — best F1 on test if we'd re-tuned threshold (NOT a valid metric to report as final)
print(f"  (For reference: if threshold were re-tuned on test, F1 best = "
      f"{test_metrics['best_f1']:.4f} @ thr={test_metrics['best_thr']:.4f} — not used)")

# Compare with calibrated model from previous step
final = joblib.load('final_model.joblib')
cal = final['model']
cal_thr = final['threshold_val_tuned']
cal_metrics = eval_at_threshold(cal, X_test, y_test, thr_to_use=cal_thr)

print('\n========== TEST: Tuned (uncalibrated) vs Calibrated ==========')
comp = pd.DataFrame([
    {'Model': 'Tuned XGB (uncal)',
     'ROC-AUC': test_metrics['roc_auc'], 'PR-AUC': test_metrics['pr_auc'],
     'Brier': test_metrics['brier'], 'F1 @ val-thr': test_metrics['f1_at_used_thr'],
     'TP': test_metrics['TP'], 'FP': test_metrics['FP'], 'FN': test_metrics['FN']},
    {'Model': 'Calibrated Isotonic',
     'ROC-AUC': cal_metrics['roc_auc'], 'PR-AUC': cal_metrics['pr_auc'],
     'Brier': cal_metrics['brier'], 'F1 @ val-thr': cal_metrics['f1_at_used_thr'],
     'TP': cal_metrics['TP'], 'FP': cal_metrics['FP'], 'FN': cal_metrics['FN']},
])
for c in ['ROC-AUC', 'PR-AUC', 'Brier', 'F1 @ val-thr']:
    comp[c] = comp[c].round(4)
print(comp.to_string(index=False))
