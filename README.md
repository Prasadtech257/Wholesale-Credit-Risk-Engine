# Institutional Wholesale Credit Risk Engine: End-to-End IFRS 9 ECL Estimation
An institutional-grade portfolio risk simulation and stress-testing engine designed to bridge regulatory credit guidelines with technical machine learning execution. 

## Key Framework Features
* **Probability of Default (PD) Modeling**: Leverages supervised machine learning (Random Forest) trained on corporate financial health metrics (Debt-to-Equity, ICR, EBITDA Margin, Current Ratio) to assign statistical 1-year baseline default probabilities across a wholesale portfolio.
* **IFRS 9 / CECL Multi-Stage Migration**: Implements structural portfolio gating rules to track Significant Increase in Credit Risk (SICR), executing automated migrations of corporate accounts from Stage 1 (12-month ECL) to Stage 2 (Lifetime ECL) or Stage 3 (Defaulted).
* **Macroeconomic Stress Testing**: Applies programmatic shock multipliers to simulate systemic economic contractions (e.g., inflation spikes or central bank interest rate hikes) to evaluate total expected credit loss volatility and capital reserves under pressure.

## Technology Stack
* **Language**: Python 3
* **Libraries**: Pandas, NumPy, Scikit-Learn
* **Framework Design**: Basel-compliant risk engine simulation

## Quick Start
Run the core analytical script to output baseline vs. stressed capital requirements:
```bash
python credit_risk_engine.py
```
