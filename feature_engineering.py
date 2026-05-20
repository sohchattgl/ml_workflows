import numpy as np
import pandas as pd

df = pd.read_csv('data_cleaned.csv')
print(f'Input: {df.shape}')

EPS = 1.0  # safe-divide epsilon for count denominators


def safe_div(num, den, eps=EPS):
    return num / den.where(den > 0, eps)


# --- Affordability ratios ---
df['loan_to_income'] = df['loan_amount'] / df['combined_annual_income']
monthly_installment = df['loan_amount'] / df['term']
monthly_income = df['combined_annual_income'] / 12.0
df['installment_to_income'] = monthly_installment / monthly_income

# --- Credit utilization ---
df['credit_utilization_ratio'] = safe_div(df['total_credit_utilized'], df['total_credit_limit'])

# --- Account composition ---
df['open_credit_ratio'] = safe_div(df['open_credit_lines'], df['total_credit_lines'])
df['cc_balance_ratio'] = safe_div(df['num_cc_carrying_balance'], df['num_open_cc_accounts'])
df['open_cc_ratio'] = safe_div(df['num_open_cc_accounts'], df['num_total_cc_accounts'])

# --- Borrowing velocity & credit age ---
df['recent_account_ratio'] = safe_div(df['accounts_opened_24m'], df['total_credit_lines'])
df['accounts_per_credit_year'] = safe_div(df['total_credit_lines'], df['credit_history_years'])
df['inquiries_per_credit_year'] = safe_div(df['inquiries_last_12m'], df['credit_history_years'])

# --- Negative history aggregate ---
df['total_negative_marks'] = (
    df['delinq_2y']
    + df['num_collections_last_12m']
    + df['num_historical_failed_to_pay']
    + df['tax_liens']
    + df['public_record_bankrupt']
)
df['has_delinq_history'] = (df['total_negative_marks'] > 0).astype(int)

# --- US Census region from state (lower-cardinality option) ---
REGION_MAP = {
    # Northeast
    **dict.fromkeys(['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'], 'Northeast'),
    # Midwest
    **dict.fromkeys(['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD'], 'Midwest'),
    # South
    **dict.fromkeys(
        ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'DC',
         'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX'],
        'South',
    ),
    # West
    **dict.fromkeys(
        ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA'],
        'West',
    ),
}
df['region'] = df['state'].map(REGION_MAP).fillna('Other')

# --- Log transforms (linear-model friendly) ---
df['log_annual_income'] = np.log1p(df['combined_annual_income'])
df['log_loan_amount'] = np.log1p(df['loan_amount'])

# --- Move target to last column ---
target = df.pop('target_delinquent')
df['target_delinquent'] = target

# --- Validate ---
assert df.isnull().sum().sum() == 0, 'NaNs introduced'
assert not np.isinf(df.select_dtypes(include=np.number).to_numpy()).any(), 'inf introduced'

df.to_csv('data_features.csv', index=False)
print(f'Output: {df.shape}')
print('Saved -> data_features.csv')
print()

new_cols = [
    'loan_to_income', 'installment_to_income',
    'credit_utilization_ratio',
    'open_credit_ratio', 'cc_balance_ratio', 'open_cc_ratio',
    'recent_account_ratio', 'accounts_per_credit_year', 'inquiries_per_credit_year',
    'total_negative_marks', 'has_delinq_history',
    'log_annual_income', 'log_loan_amount',
]
print('Engineered numeric features (describe):')
print(df[new_cols].describe().T[['mean', 'std', 'min', '50%', 'max']].round(3))
print()
print('Region distribution:')
print(df['region'].value_counts())
print()
print('Correlation of new features with target:')
corrs = df[new_cols + ['target_delinquent']].corr()['target_delinquent'].drop('target_delinquent')
print(corrs.sort_values(key=abs, ascending=False).round(4))
