import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET = 'target_delinquent'
CATEGORICAL_COLS = ['state', 'homeownership', 'loan_purpose', 'application_type', 'region']

train = pd.read_csv('data_train.csv')
val = pd.read_csv('data_val.csv')
test = pd.read_csv('data_test.csv')
print(f'Loaded: train={train.shape}  val={val.shape}  test={test.shape}')

numeric_cols = [c for c in train.columns if c not in CATEGORICAL_COLS + [TARGET]]
print(f'Numeric features to scale: {len(numeric_cols)}')
print(f'Categorical features (left raw): {len(CATEGORICAL_COLS)}')

# Fit on TRAIN ONLY, transform all three splits
scaler = StandardScaler()
train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
val[numeric_cols] = scaler.transform(val[numeric_cols])
test[numeric_cols] = scaler.transform(test[numeric_cols])

# Save scaled splits (overwriting the unscaled versions) + scaler artifact
train.to_csv('data_train.csv', index=False)
val.to_csv('data_val.csv', index=False)
test.to_csv('data_test.csv', index=False)
joblib.dump({'scaler': scaler, 'numeric_cols': numeric_cols}, 'scaler.joblib')

print('\nSaved -> data_train.csv, data_val.csv, data_test.csv, scaler.joblib')

# Sanity checks
print('\nTrain numeric sanity (should be mean~0, std~1):')
print(train[numeric_cols[:5]].agg(['mean', 'std']).round(4))
print('\nVal numeric sanity (mean/std drift from 0/1 expected — using train stats):')
print(val[numeric_cols[:5]].agg(['mean', 'std']).round(4))
print('\nTest numeric sanity:')
print(test[numeric_cols[:5]].agg(['mean', 'std']).round(4))

print('\nCategoricals untouched (sample from train):')
print(train[CATEGORICAL_COLS].head(3))
