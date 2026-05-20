import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

TARGET = 'target_delinquent'
OHE_COLS = ['homeownership', 'application_type', 'region', 'loan_purpose']
TE_COL = 'state'
SMOOTHING_ALPHA = 10   # Empirical-Bayes smoothing strength
N_FOLDS = 5            # OOF folds for target encoding
SEED = 42

train = pd.read_csv('data_train.csv')
val = pd.read_csv('data_val.csv')
test = pd.read_csv('data_test.csv')
print(f'Loaded: train={train.shape}  val={val.shape}  test={test.shape}')

# ---------- One-Hot Encoding ----------
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float64)
ohe.fit(train[OHE_COLS])
ohe_feature_names = ohe.get_feature_names_out(OHE_COLS).tolist()

def apply_ohe(df):
    arr = ohe.transform(df[OHE_COLS])
    ohe_df = pd.DataFrame(arr, columns=ohe_feature_names, index=df.index)
    return pd.concat([df.drop(columns=OHE_COLS), ohe_df], axis=1)

train = apply_ohe(train)
val = apply_ohe(val)
test = apply_ohe(test)
print(f'OHE added {len(ohe_feature_names)} columns')

# ---------- Target Encoding for `state` (smoothed, OOF for train) ----------
def smoothed_mean(counts, means, global_mean, alpha):
    """Empirical-Bayes blend: lean toward global_mean when counts are small."""
    return (counts * means + alpha * global_mean) / (counts + alpha)

y_train = train[TARGET].values
global_mean = float(y_train.mean())

# OOF encoding for TRAIN (each fold uses only the other folds' stats)
oof_encoded = np.zeros(len(train))
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for tr_idx, ho_idx in skf.split(train, y_train):
    fold_tr = train.iloc[tr_idx]
    fold_global = fold_tr[TARGET].mean()
    stats = fold_tr.groupby(TE_COL)[TARGET].agg(['count', 'mean'])
    fold_map = smoothed_mean(stats['count'], stats['mean'], fold_global, SMOOTHING_ALPHA)
    oof_encoded[ho_idx] = (
        train.iloc[ho_idx][TE_COL].map(fold_map).fillna(fold_global).values
    )

# Final mapping from FULL train (used for val/test)
full_stats = train.groupby(TE_COL)[TARGET].agg(['count', 'mean'])
state_encoding_map = smoothed_mean(
    full_stats['count'], full_stats['mean'], global_mean, SMOOTHING_ALPHA
).to_dict()

train['state_encoded'] = oof_encoded
val['state_encoded'] = val[TE_COL].map(state_encoding_map).fillna(global_mean)
test['state_encoded'] = test[TE_COL].map(state_encoding_map).fillna(global_mean)

# Drop raw state column
train.drop(columns=[TE_COL], inplace=True)
val.drop(columns=[TE_COL], inplace=True)
test.drop(columns=[TE_COL], inplace=True)

# Move target to last column
def target_last(df):
    t = df.pop(TARGET)
    df[TARGET] = t
    return df
train, val, test = target_last(train), target_last(val), target_last(test)

# Save
train.to_csv('data_train.csv', index=False)
val.to_csv('data_val.csv', index=False)
test.to_csv('data_test.csv', index=False)

joblib.dump({
    'ohe': ohe,
    'ohe_cols': OHE_COLS,
    'ohe_feature_names': ohe_feature_names,
    'state_encoding_map': state_encoding_map,
    'state_global_mean': global_mean,
    'smoothing_alpha': SMOOTHING_ALPHA,
}, 'encoders.joblib')

print(f'\nFinal shapes: train={train.shape}  val={val.shape}  test={test.shape}')
print('Saved -> data_train.csv, data_val.csv, data_test.csv, encoders.joblib')

# ---------- Diagnostics ----------
print(f'\nGlobal positive rate (train): {global_mean:.4f}')
print(f'\nTop 5 RISKIEST states (highest smoothed mean) — from train:')
ranked = pd.Series(state_encoding_map).sort_values(ascending=False)
counts = full_stats['count']
top = pd.DataFrame({'encoded': ranked.head(5), 'train_n': counts.loc[ranked.head(5).index]})
print(top.round(4))
print(f'\nTop 5 SAFEST states:')
bot = pd.DataFrame({'encoded': ranked.tail(5), 'train_n': counts.loc[ranked.tail(5).index]})
print(bot.round(4))

print(f'\nOOF state_encoded stats (train):')
print(train['state_encoded'].describe().round(4))
