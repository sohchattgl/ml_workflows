import json
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
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
test = pd.read_csv('data_test.csv')

X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
X_val, y_val     = val.drop(columns=[TARGET]),   val[TARGET]
X_test, y_test   = test.drop(columns=[TARGET]),  test[TARGET]
print(f'Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}')

# ---------- Recreate the tuned base estimator from saved params ----------
with open('xgb_best_params.json') as f:
    raw = json.load(f)
best_params = {}
int_keys = {'n_estimators', 'max_depth', 'min_child_weight'}
for k, v in raw.items():
    try:
        best_params[k] = int(round(float(v))) if k in int_keys else float(v)
    except (ValueError, TypeError):
        best_params[k] = v

n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = n_neg / n_pos

def make_base():
    return XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        tree_method='hist',
        scale_pos_weight=spw,
        random_state=SEED,
        n_jobs=-1,
        **best_params,
    )

# ---------- Calibrate (two methods, pick by val Brier score) ----------
def fit_calibrated(method):
    cal = CalibratedClassifierCV(
        estimator=make_base(),
        method=method,
        cv=5,
        n_jobs=-1,
    )
    cal.fit(X_train, y_train)
    return cal

print('\nFitting calibrated models (cv=5)...')
cal_iso = fit_calibrated('isotonic')
cal_sig = fit_calibrated('sigmoid')

# Uncalibrated reference (the tuned XGBoost, refit fresh for fairness)
print('Fitting uncalibrated reference...')
uncal = make_base()
uncal.fit(X_train, y_train)


def all_metrics(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    auroc = roc_auc_score(y, proba)
    ap = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    prec, rec, thr = precision_recall_curve(y, proba)
    f1c = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    bi = int(np.nanargmax(f1c[:-1]))
    return {
        'roc_auc': auroc, 'pr_auc': ap, 'brier': brier,
        'f1_best': f1c[bi], 'thr_best': thr[bi],
        'precision_at_best': prec[bi], 'recall_at_best': rec[bi],
        'proba': proba,
    }


print('\n========== VAL: pick calibration method ==========')
val_rows = []
for name, m in [('Uncalibrated', uncal), ('Isotonic', cal_iso), ('Sigmoid (Platt)', cal_sig)]:
    r = all_metrics(m, X_val, y_val)
    val_rows.append({'Model': name, 'ROC-AUC': r['roc_auc'], 'PR-AUC': r['pr_auc'],
                     'Brier': r['brier'], 'F1 best': r['f1_best'], 'Thr': r['thr_best']})
val_df = pd.DataFrame(val_rows)
for c in ['ROC-AUC', 'PR-AUC', 'Brier', 'F1 best', 'Thr']:
    val_df[c] = val_df[c].round(4)
print(val_df.to_string(index=False))

# Pick winner by Brier (lower = better calibrated)
cal_metrics_val = {'Isotonic': all_metrics(cal_iso, X_val, y_val),
                   'Sigmoid (Platt)': all_metrics(cal_sig, X_val, y_val)}
winner_name = min(cal_metrics_val, key=lambda k: cal_metrics_val[k]['brier'])
winner = cal_iso if winner_name == 'Isotonic' else cal_sig
print(f'\nChosen calibrator: {winner_name} (lowest val Brier)')

# ---------- Calibration curve diagnostic on val ----------
print(f'\n{winner_name} calibration curve on val (10 bins):')
prob_true, prob_pred = calibration_curve(y_val, cal_metrics_val[winner_name]['proba'],
                                          n_bins=10, strategy='quantile')
cc = pd.DataFrame({'pred_prob_mean': prob_pred, 'actual_pos_rate': prob_true}).round(4)
print(cc.to_string(index=False))

# ---------- FINAL TEST EVALUATION ----------
print('\n' + '=' * 60)
print('FINAL TEST EVALUATION — chosen model: XGBoost + ' + winner_name)
print('=' * 60)

test_uncal = all_metrics(uncal, X_test, y_test)
test_cal = all_metrics(winner, X_test, y_test)

# Use the threshold tuned on VAL (not test — to honor the "test-once" pattern)
val_thr_uncal = all_metrics(uncal, X_val, y_val)['thr_best']
val_thr_cal = cal_metrics_val[winner_name]['thr_best']

def report_test(label, model, X, y, thr):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= thr).astype(int)
    auroc = roc_auc_score(y, proba)
    ap = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    f1 = f1_score(y, pred, zero_division=0)
    cm = confusion_matrix(y, pred)
    print(f'\n--- {label} ---')
    print(f'  ROC-AUC : {auroc:.4f}')
    print(f'  PR-AUC  : {ap:.4f}  (random baseline = {y.mean():.4f})')
    print(f'  Brier   : {brier:.4f}')
    print(f'  F1 @ val-tuned threshold ({thr:.4f}): {f1:.4f}')
    print(f'  Confusion matrix:')
    print(f'    TN={cm[0,0]:>5}  FP={cm[0,1]:>4}')
    print(f'    FN={cm[1,0]:>5}  TP={cm[1,1]:>4}')
    return {'ROC-AUC': auroc, 'PR-AUC': ap, 'Brier': brier, 'F1': f1,
            'TP': cm[1,1], 'FP': cm[0,1], 'FN': cm[1,0], 'TN': cm[0,0]}

print('\nThresholds chosen on VAL and applied (not re-tuned) on TEST.')
r_uncal = report_test('Uncalibrated (XGBoost tuned)', uncal, X_test, y_test, val_thr_uncal)
r_cal = report_test(f'Calibrated ({winner_name})', winner, X_test, y_test, val_thr_cal)

print('\n========== TEST SUMMARY ==========')
summary = pd.DataFrame([
    {'Model': 'Uncalibrated', **r_uncal},
    {'Model': f'Calibrated ({winner_name})', **r_cal},
])
for c in ['ROC-AUC', 'PR-AUC', 'Brier', 'F1']:
    summary[c] = summary[c].round(4)
print(summary.to_string(index=False))

# Test-set calibration curve
print(f'\nTest-set calibration curve (10 bins) for {winner_name}:')
prob_true_t, prob_pred_t = calibration_curve(y_test, test_cal['proba'],
                                              n_bins=10, strategy='quantile')
cc_test = pd.DataFrame({'pred_prob_mean': prob_pred_t, 'actual_pos_rate': prob_true_t}).round(4)
print(cc_test.to_string(index=False))

# Persist final
joblib.dump({
    'model': winner,
    'calibration_method': winner_name,
    'threshold_val_tuned': val_thr_cal,
    'features': list(X_train.columns),
}, 'final_model.joblib')
print('\nSaved -> final_model.joblib')
