import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1. GENERATE SYNTHETIC WHOLESALE PORTFOLIO DATA
np.random.seed(42)
n_borrowers = 2000

data = {
    'Debt_to_Equity': np.random.uniform(0.5, 4.5, n_borrowers),
    'Interest_Coverage': np.random.uniform(0.8, 8.0, n_borrowers),
    'Current_Ratio': np.random.uniform(0.7, 2.5, n_borrowers),
    'EBITDA_Margin': np.random.uniform(0.05, 0.40, n_borrowers),
    'EAD': np.random.uniform(5000000, 50000000, n_borrowers)  # Exposure at Default in USD/INR
}

df = pd.DataFrame(data)

# Define a logical default trigger based on weak financials
default_score = (
    (df['Debt_to_Equity'] > 3.0).astype(int) * 0.4 +
    (df['Interest_Coverage'] < 1.5).astype(int) * 0.4 +
    (df['EBITDA_Margin'] < 0.10).astype(int) * 0.2 +
    np.random.normal(0, 0.1, n_borrowers)
)
df['Default'] = (default_score > 0.45).astype(int)

# 2. MODEL TRAINING (Predicting Probability of Default - PD)
X = df[['Debt_to_Equity', 'Interest_Coverage', 'Current_Ratio', 'EBITDA_Margin']]
y = df['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Model Performance
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
print(f"Base Model ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Assign Baseline 1-Year PD to entire portfolio
df['Baseline_PD'] = rf_model.predict_proba(X)[:, 1]

# 3. MACROECONOMIC STRESS TESTING SCENARIO
# Simulate severe GDP contraction / interest rate shock multiplying default likelihood
stress_multiplier = 1.45 
df['Stressed_PD'] = np.clip(df['Baseline_PD'] * stress_multiplier, 0, 1)

# 4. IFRS 9 ECL STAGE CLASSIFICATION & CALCULATION
# Stage 1: Normal (12-month ECL), Stage 2: Significant Increase in Credit Risk (Lifetime ECL proxy), Stage 3: Defaulted
def assign_stage(row):
    if row['Stressed_PD'] > 0.40:
        return 'Stage 3 (Default)'
    elif row['Stressed_PD'] > 0.20:
        return 'Stage 2 (SICR)'
    else:
        return 'Stage 1 (Normal)'

df['IFRS9_Stage'] = df.apply(assign_stage, axis=1)

# Loss Given Default (LGD) assumed at a standard 45% for unsecured wholesale corporate loans
LGD = 0.45
df['ECL_Baseline'] = df['EAD'] * df['Baseline_PD'] * LGD
df['ECL_Stressed'] = df['EAD'] * df['Stressed_PD'] * LGD

# 5. PORTFOLIO SUMMARY OUTPUT
print("\n--- Portfolio Stress Test Results ---")
print(df['IFRS9_Stage'].value_counts())
print(f"\nTotal Baseline Portfolio ECL: ${df['ECL_Baseline'].sum():,.2f}")
print(f"Total Stressed Portfolio ECL: ${df['ECL_Stressed'].sum():,.2f}")
