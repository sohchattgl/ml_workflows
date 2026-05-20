"""
Calibrate the EXACT saved tuned XGBoost via prefit-style calibration on val.

Manual calibration (since sklearn 1.8 removed cv='prefit'):
- Isotonic: IsotonicRegression on (val_proba, val_y)
- Sigmoid:  LogisticRegression on (val_proba_logit, val_y) — i.e., Platt scaling
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

TARGET = 'target_delinquent'

val = pd.read_csv('data_val.csv')
test = pd.read_csv('data_test.csv')
X_val, y_val = val.drop(columns=[TARGET]), val[TARGET]
X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

tuned = joblib.load('model_xgb_tuned.joblib')

# Raw predictions on val + test from the prefit tuned model
proba_val_raw = tuned.predict_proba(X_val)[:, 1]
proba_test_raw = tuned.predict_proba(X_test)[:, 1]

# ---------- Fit calibrators on (val_proba, val_y) ----------
iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
iso.fit(proba_val_raw, y_val)

# Platt scaling = logistic regression on the model's logits.
# Use logit(proba) as input so the calibrator is a true sigmoid mapping.
def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

platt = LogisticRegression(C=1e6, solver='lbfgs')   # ~unregularized
platt.fit(logit(proba_val_raw).reshape(-1, 1), y_val)

# ---------- Apply calibrators ----------
proba_val_iso = iso.predict(proba_val_raw)
proba_val_sig = platt.predict_proba(logit(proba_val_raw).reshape(-1, 1))[:, 1]
proba_test_iso = iso.predict(proba_test_raw)
proba_test_sig = platt.predict_proba(logit(proba_test_raw).reshape(-1, 1))[:, 1]


def report(label, proba, y, thr=None):
    """If thr is None, pick best-F1 threshold and return it."""
    auroc = roc_auc_score(y, proba)
    ap = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    prec, rec, t = precision_recall_curve(y, proba)
    f1c = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    bi = int(np.nanargmax(f1c[:-1]))
    if thr is None:
        thr_used = t[bi]
        f1_best = f1c[bi]
    else:
        thr_used = thr
        pred = (proba >= thr).astype(int)
        f1_best = f1_score(y, pred, zero_division=0)
    pred = (proba >= thr_used).astype(int)
    cm = confusion_matrix(y, pred)
    return {
        'Model': label, 'ROC-AUC': auroc, 'PR-AUC': ap, 'Brier': brier,
        'F1': f1_best, 'thr': thr_used,
        'TP': cm[1,1], 'FP': cm[0,1], 'FN': cm[1,0], 'TN': cm[0,0],
    }


# Pick best-F1 thresholds on VAL (note: val is also where the calibrator was trained,
# so val metrics are optimistic — test is the honest read).
val_results = [
    report('Tuned (uncal)',          proba_val_raw, y_val),
    report('Tuned + Isotonic',       proba_val_iso, y_val),
    report('Tuned + Sigmoid (Platt)', proba_val_sig, y_val),
]
val_df = pd.DataFrame(val_results)
print('========== VAL (calibrator was fit on val — optimistic) ==========')
for c in ['ROC-AUC', 'PR-AUC', 'Brier', 'F1', 'thr']:
    val_df[c] = val_df[c].round(4)
print(val_df.to_string(index=False))

# Now apply val-tuned thresholds to TEST (honest evaluation)
thr_uncal = val_results[0]['thr']
thr_iso   = val_results[1]['thr']
thr_sig   = val_results[2]['thr']

test_results = [
    report('Tuned (uncal)',          proba_test_raw, y_test, thr=thr_uncal),
    report('Tuned + Isotonic',       proba_test_iso, y_test, thr=thr_iso),
    report('Tuned + Sigmoid (Platt)', proba_test_sig, y_test, thr=thr_sig),
]
test_df = pd.DataFrame(test_results)
print('\n========== TEST (val-tuned thresholds applied unchanged) ==========')
for c in ['ROC-AUC', 'PR-AUC', 'Brier', 'F1', 'thr']:
    test_df[c] = test_df[c].round(4)
print(test_df.to_string(index=False))

# Calibration curves on test
print('\n----- Test calibration curves (10 quantile bins) -----')
for label, proba in [
    ('Uncalibrated',     proba_test_raw),
    ('Isotonic',         proba_test_iso),
    ('Sigmoid (Platt)',  proba_test_sig),
]:
    pt, pp = calibration_curve(y_test, proba, n_bins=10, strategy='quantile')
    cc = pd.DataFrame({'pred_prob_mean': pp, 'actual_pos_rate': pt}).round(4)
    print(f'\n{label}:')
    print(cc.to_string(index=False))

# Persist final calibrated model (saving both calibrators for downstream choice)
joblib.dump({
    'base_model': tuned,
    'isotonic': iso, 'isotonic_thr_val': thr_iso,
    'platt': platt, 'platt_thr_val': thr_sig,
    'features': list(X_val.columns),
}, 'model_tuned_xgb_calibrated_prefit.joblib')
print('\nSaved -> model_tuned_xgb_calibrated_prefit.joblib')
