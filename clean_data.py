import pandas as pd
import numpy as np

df = pd.read_csv('data_without_leakage.csv')
print(f'Input: {df.shape}')

# 1. Drop zero-variance column (only 0 or NaN)
df = df.drop(columns=['num_accounts_120d_past_due'])

# 2. Unify joint vs individual income / DTI
df['combined_annual_income'] = np.where(
    df['application_type'].eq('joint'),
    df['annual_income_joint'],
    df['annual_income'],
)
df['combined_dti'] = np.where(
    df['application_type'].eq('joint'),
    df['debt_to_income_joint'],
    df['debt_to_income'],
)
df = df.drop(columns=['annual_income_joint', 'debt_to_income_joint'])

# 3. MNAR months_since_* — NaN means "never". Indicator + sentinel fill.
NEVER_SENTINEL = 999
for col in [
    'months_since_last_delinq',
    'months_since_90d_late',
    'months_since_last_credit_inquiry',
]:
    df[f'{col}_never'] = df[col].isnull().astype(int)
    df[col] = df[col].fillna(NEVER_SENTINEL)

# 4. emp_length — median impute + missingness indicator
df['emp_length_missing'] = df['emp_length'].isnull().astype(int)
df['emp_length'] = df['emp_length'].fillna(df['emp_length'].median())

# 5. debt_to_income — median impute (24 rows)
df['debt_to_income'] = df['debt_to_income'].fillna(df['debt_to_income'].median())
# combined_dti inherits any residual NaN from individual rows above — re-fill
df['combined_dti'] = df['combined_dti'].fillna(df['combined_dti'].median())

# 6. earliest_credit_line -> credit_history_years (baseline = max year in data + 1)
baseline = int(df['earliest_credit_line'].max()) + 1  # 2016
df['credit_history_years'] = baseline - df['earliest_credit_line']
df = df.drop(columns=['earliest_credit_line'])

# Reorder so target is last
target = df.pop('target_delinquent')
df['target_delinquent'] = target

# Assertions
assert df.isnull().sum().sum() == 0, 'NaNs remain'
assert df.duplicated().sum() == 0, 'duplicates present'

df.to_csv('data_cleaned.csv', index=False)
print(f'Output: {df.shape}')
print(f'Saved -> data_cleaned.csv')
print()
print('Final dtypes:')
print(df.dtypes.value_counts())
print()
print('Sanity check on engineered columns:')
print(df[['combined_annual_income','combined_dti','credit_history_years',
         'months_since_last_delinq_never','emp_length_missing']].describe())
