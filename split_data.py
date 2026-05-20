import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data_features.csv')
print(f'Input: {df.shape}')

TARGET = 'target_delinquent'
SEED = 42

y = df[TARGET]
X = df.drop(columns=[TARGET])

# Two-stage stratified split:
#   Stage 1: hold out 20% as test
#   Stage 2: from remaining 80%, split into train (70/80 = 87.5%) and val (10/80 = 12.5%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED,
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=SEED,
)

# Reattach target and save
for name, X_part, y_part in [
    ('train', X_train, y_train),
    ('val',   X_val,   y_val),
    ('test',  X_test,  y_test),
]:
    out = X_part.copy()
    out[TARGET] = y_part.values
    out.to_csv(f'data_{name}.csv', index=False)

# Verify sizes & class balance
print(f'\n{"Split":<6} {"Rows":>6} {"%":>7} {"Pos":>5} {"Pos%":>7}')
total = len(df)
for name, y_part in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    n = len(y_part)
    print(f'{name:<6} {n:>6} {n/total:>6.1%} {y_part.sum():>5} {y_part.mean():>6.2%}')
print(f'{"TOTAL":<6} {total:>6} {1.0:>6.1%} {y.sum():>5} {y.mean():>6.2%}')

print('\nSaved -> data_train.csv, data_val.csv, data_test.csv')
