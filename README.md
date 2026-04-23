# 🏦 Loan Eligibility & Risk Analysis
### A Financial Data Analytics Project by Lesson Shepherd Karidza

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Stage 1 — Data Loading & Understanding](#stage-1--data-loading--understanding)
4. [Stage 2 — Data Quality Assessment](#stage-2--data-quality-assessment)
5. [Stage 3 — Exploratory Data Analysis (EDA)](#stage-3--exploratory-data-analysis-eda)
6. [Stage 4 — Feature Engineering](#stage-4--feature-engineering)
7. [Stage 5 — Correlation & Statistical Analysis](#stage-5--correlation--statistical-analysis)
8. [Key Findings Summary](#key-findings-summary)
9. [Tools & Technologies](#tools--technologies)
10. [Next Steps — Modelling](#next-steps--modelling)

---

## Project Overview

This project performs an end-to-end financial data analytics pipeline on a merged borrower and loan application dataset. The goal is to:

- Understand the profile of borrowers who default on loans
- Engineer meaningful financial risk features (Risk Score, Income Segment, Loan Eligibility)
- Identify the strongest statistical predictors of loan default
- Build a foundation for a machine learning loan default prediction model

This analysis is directly applicable to **credit risk**, **loan underwriting**, and **financial risk management** use cases in banking and fintech environments.

---

## Dataset Description

The dataset was created by joining two tables:
- `borrower_profiles` — demographic and financial profile of each borrower
- `loan_applications` — loan-specific details per application

**Final merged dataset: 601 rows × 22 columns**

| Column | Type | Description |
|---|---|---|
| `borrower_id` | ID | Unique borrower identifier |
| `age` | Numerical | Borrower age (20–70) |
| `state` | Categorical | US state of residence |
| `education_level` | Categorical | Highest education attained |
| `employment_status` | Categorical | Employment type |
| `years_employed` | Numerical | Years in current employment |
| `annual_income` | Numerical | Gross annual income (USD) |
| `credit_score` | Numerical | Credit score (550–850) |
| `home_ownership` | Categorical | Rent / Mortgage / Own |
| `dependents` | Numerical | Number of dependents |
| `existing_monthly_debt` | Numerical | Current monthly debt obligations |
| `loan_id` | ID | Unique loan identifier |
| `application_date` | Date | Date of loan application |
| `loan_purpose` | Categorical | Reason for the loan |
| `loan_amount` | Numerical | Loan amount requested (USD) |
| `term_months` | Numerical | Loan repayment term |
| `interest_rate` | Numerical | Annual interest rate (%) |
| `monthly_payment` | Numerical | Monthly repayment amount |
| `dti_ratio` | Numerical | Debt-to-income ratio (%) |
| `loan_status` | Categorical | Current / Default / Late / Paid Off |
| `days_delinquent` | Numerical | Days past due |
| `defaulted` | Binary | Target variable (1 = Default, 0 = No Default) |

---

## Stage 1 — Data Loading & Understanding

### Code
```python
import pandas as pd
import numpy as np

df = pd.read_csv("your_file.csv")

print("Shape:", df.shape)
print("\nColumns & Types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())
print("\nStatistics:")
print(df.describe())
```

### Findings
- Dataset contains **601 rows and 22 columns** after joining both tables
- Mix of numerical, categorical, date, and binary columns
- Target variable `defaulted` is binary (0 = No Default, 1 = Default)
- No immediate structural issues detected on first inspection

### ✅ Actions Taken
| Action | Reason |
|---|---|
| Confirmed column names and data types | Ensures correct handling of numerical vs categorical features in later stages |
| Checked target variable (`defaulted`) distribution | Establishes baseline class balance before modelling |
| Reviewed descriptive statistics | Identified range, mean, and potential outliers for all numerical columns |
| Confirmed dataset size (601 rows) | Informs model selection — small dataset rules out deep learning; limits to classical ML |

### 💡 Business Implication
> A 601-row dataset is sufficient for exploratory analysis and classical ML models. Results should be treated as indicative and validated on a larger population before production deployment.

---

## Stage 2 — Data Quality Assessment

### Code
```python
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(df.nunique().sort_values())
print(f"Duplicate rows: {df.duplicated().sum()}")
```

### Findings

| Check | Result |
|---|---|
| Missing values | ✅ None detected |
| Duplicate rows | ✅ None detected |
| Data types | ✅ All appropriate |
| Outliers | ⚠️ `dti_ratio` has extreme values (up to 180%) |
| `days_delinquent` | ⚠️ Heavily zero-inflated (~400 rows = 0) |

**Category distributions:**

| Column | Categories |
|---|---|
| `education_level` | Bachelor (229), Master (148), High School (106), Associate (98), Doctorate (20) |
| `employment_status` | Full-Time (305), Self-Employed (105), Contract (66), Part-Time (65), Retired (60) |
| `home_ownership` | Mortgage (262), Rent (210), Own (129) |
| `loan_purpose` | 10 categories, evenly distributed ~51–70 per category |
| `loan_status` | Current (239), Default (146), Paid Off (113), Late (103) |

> ⚠️ **Class imbalance noted:** Only 24.4% of borrowers defaulted — addressed in modelling stage with SMOTE.

### ✅ Actions Taken
| Action | Reason |
|---|---|
| Confirmed zero missing values | No imputation strategy needed — dataset is complete |
| Confirmed zero duplicates | No deduplication needed — each row is a unique borrower-loan pair |
| Flagged `dti_ratio` outliers (up to 180%) | Capped at 100% in Feature Engineering (Stage 4) — values above 100% are financially implausible |
| Flagged `days_delinquent` zero-inflation | Converted to binary flag (`is_delinquent`) in Stage 4 — raw sparse values are not model-friendly |
| Noted class imbalance (24.4% default) | Planned SMOTE oversampling in modelling stage to prevent model bias toward majority class |
| Noted small Doctorate sample (n=20) | Flagged for caution — 30% default rate for this group may not be statistically reliable |

### 💡 Business Implication
> The dataset is clean and ready for analysis. The main risk to address is class imbalance — a model trained without correction would learn to predict "no default" almost always, achieving 75% accuracy while being useless for catching actual defaulters.

---

## Stage 3 — Exploratory Data Analysis (EDA)

### 3a — Numerical Distributions

| Feature | Distribution Shape | Notes |
|---|---|---|
| `age` | Roughly uniform (20–70) | No strong age concentration |
| `annual_income` | Right-skewed | Most borrowers earn $30K–$80K; one outlier near $160K |
| `credit_score` | Roughly uniform (550–850) | Spike at 850 (maximum score) |
| `years_employed` | Right-skewed | Most borrowers employed 0–10 years |
| `loan_amount` | Right-skewed | Most loans between $5K–$20K |
| `interest_rate` | Bell-shaped | Centred around 9–11% |
| `monthly_payment` | Right-skewed | Most payments under $1,000 |
| `dti_ratio` | Bell-shaped with outliers | Bulk 20–65%, dangerous outliers at 140–180% |
| `days_delinquent` | Zero-inflated | ~400 borrowers at 0; sparse beyond 50 days |
| `existing_monthly_debt` | Right-skewed | Most under $2,500/month |

### ✅ Actions Taken — Distributions
| Action | Reason |
|---|---|
| Identified right-skewed features | StandardScaler applied in pipeline to normalise — prevents linear models from being biased by scale |
| Confirmed `interest_rate` is bell-shaped | Normally distributed — no transformation needed; safe to use directly in models |
| Noted `credit_score` spike at 850 | Ceiling effect — not an error; confirms many borrowers hit the maximum score |
| Identified `dti_ratio` outliers (140–180%) | Capped at 100% in Feature Engineering — values above 100% are financially impossible |
| Confirmed `days_delinquent` zero-inflation | Converted to binary `is_delinquent` flag — raw values too sparse for meaningful modelling |

---

### 3b — Loan Status Distribution

| Status | Count | % of Total |
|---|---|---|
| Current | 239 | 39.8% |
| Default | 146 | 24.3% |
| Paid Off | 113 | 18.8% |
| Late | 103 | 17.1% |

> ⚠️ Combined Default + Late = **41.4%** of the portfolio — a significant at-risk segment.

### ✅ Actions Taken — Loan Status
| Action | Reason |
|---|---|
| Confirmed target variable distribution | 24.3% default rate — moderate imbalance; SMOTE planned for modelling |
| Flagged Late loans (103 borrowers) as at-risk | Pre-default candidates — recommended for proactive intervention |
| Noted Current + Paid Off = 58.6% of portfolio | Majority healthy — model should focus on identifying the 41.4% risk segment |

### 💡 Business Implication
> The **Late** segment (103 borrowers) represents the most actionable group. A targeted retention programme — payment restructuring, grace periods, or counselling — for Late borrowers could reduce default rates before they materialise.

---

### 3c — Default Rate by Category

**By Education Level:**

| Education | Default Rate | Action |
|---|---|---|
| Doctorate | 30.0% ⚠️ | Treat with caution — small sample (n=20), not statistically reliable |
| High School | 27.4% 🔴 | Flag as higher risk — consider stricter DTI or collateral requirements |
| Bachelor | 26.6% 🔴 | Moderate risk — evaluate alongside income and DTI |
| Master | 20.9% 🟡 | Lower risk — favourable signal in credit scoring |
| Associate | 19.4% ✅ | Lowest default rate — positive underwriting signal |

### ✅ Actions Taken — Education
| Action | Reason |
|---|---|
| Included `education_level` as a model feature | Meaningful variation across groups — useful categorical predictor |
| Excluded Doctorate from standalone business rules | Sample too small (n=20) — individual model prediction more reliable than group-level rules |

---

**By Employment Status:**

| Employment | Default Rate | Action |
|---|---|---|
| Part-Time | 27.7% 🔴 | Highest risk — require higher credit score or lower DTI threshold |
| Self-Employed | 24.8% 🟡 | Income verification required — earnings may be irregular |
| Full-Time | 23.9% 🟡 | Standard processing — baseline risk group |
| Retired | 23.3% 🟡 | Assess pension/fixed income stability before approval |
| Contract | 22.7% ✅ | Lowest default rate — contract income often stable and predictable |

### ✅ Actions Taken — Employment
| Action | Reason |
|---|---|
| Included `employment_status` as a model feature | Clear variation — part-time vs contract gap of 5% is meaningful |
| Recommended income verification for Self-Employed | Irregular income makes DTI ratio less reliable as a standalone metric |

### 💡 Business Implication
> Part-time workers should trigger a mandatory secondary review in the loan approval workflow — requiring either a co-signer, collateral, or a credit score above 700 before approval.

---

**By Home Ownership:**

| Ownership | Default Rate | Action |
|---|---|---|
| Rent | 31.0% 🔴 | Highest risk — apply stricter eligibility criteria and lower loan caps |
| Mortgage | 21.8% 🟡 | Moderate risk — standard processing |
| Own | 18.6% ✅ | Lowest risk — eligible for preferential rates and higher loan limits |

### ✅ Actions Taken — Home Ownership
| Action | Reason |
|---|---|
| Included `home_ownership` as a model feature | 12.4% gap between Rent and Own is a significant risk signal |
| Incorporated into eligibility rules | Renters with low credit scores and high DTI flagged as Not Eligible |

### 💡 Business Implication
> Renters defaulting at 31% suggests lack of asset backing. For renter applicants, reduce the maximum approvable loan amount or require a lower DTI ceiling (≤35% instead of ≤43%).

---

**By Loan Purpose:**

| Purpose | Default Rate | Action |
|---|---|---|
| Wedding | 32.1% 🔴 | Highest risk — apply premium interest rate or lower max loan amount |
| Home Improvement | 28.6% 🔴 | Elevated risk — verify income and project estimates |
| Auto Loan | 27.1% 🔴 | Moderate-high risk — asset-backed but depreciating collateral |
| Business Loan | 24.1% 🟡 | Requires business plan and revenue verification |
| Medical Expenses | 20.6% ✅ | Lowest risk — need-based, non-discretionary spending |

### ✅ Actions Taken — Loan Purpose
| Action | Reason |
|---|---|
| Included `loan_purpose` as a model feature | 11.5% range between highest and lowest default rates — strong categorical signal |
| Recommended purpose-based pricing tiers | Wedding and Home Improvement loans should carry higher rates to compensate for elevated risk |

### 💡 Business Implication
> Consider capping loan amounts for lifestyle purposes (Wedding, Vacation) at $15,000 unless credit score exceeds 700. Need-based loans (Medical, Education) can be approved at higher amounts with standard criteria.

---

### 3d — Key Financial Features vs Loan Status

| Feature | Paid Off | Default | Current | Late |
|---|---|---|---|---|
| `annual_income` median | ~$65K | ~$55K 🔴 | ~$65K | ~$60K |
| `credit_score` median | ~750 | ~630 🔴 | ~735 | ~710 |
| `interest_rate` median | ~10% | ~12% 🔴 | ~10% | ~11% |
| `dti_ratio` median | ~48% | ~57% 🔴 | ~45% | ~48% |
| `days_delinquent` median | ~0 | ~105 🔴 | ~0 | ~10 |
| `loan_amount` | Similar across all groups | | | |

### ✅ Actions Taken — Financial Features
| Action | Reason |
|---|---|
| Confirmed `credit_score` as primary separator | 120-point median gap between Default (630) and Paid Off (750) — strongest visual signal |
| Confirmed `interest_rate` as secondary signal | 2% median gap confirms lender pricing already reflects underlying default risk |
| Noted `loan_amount` shows no meaningful separation | Loan amount alone does not predict default — repayment capacity matters more than loan size |
| Excluded `days_delinquent` from approval model | Post-loan data — available only after disbursement, not at point of approval decision |
| Set credit score threshold at 650 for eligibility | Based on clear separation between Default (~630) and Current/Paid Off (~735–750) medians |

### 💡 Business Implication
> Loan amount alone is not a reliable risk indicator. A borrower's capacity to repay — measured by credit score, DTI, and income — matters far more than the raw loan size. Underwriting should prioritise repayment capacity over loan amount caps.

---

## Stage 4 — Feature Engineering

Six new features were engineered to enhance model performance and business interpretability.

### 4a — Income Segment
```python
p33 = df["annual_income"].quantile(0.33)  # $47,833
p66 = df["annual_income"].quantile(0.66)  # $72,256

df["income_segment"] = pd.cut(df["annual_income"],
    bins=[-np.inf, p33, p66, np.inf],
    labels=["Low", "Medium", "High"])
```

| Segment | Threshold | Count |
|---|---|---|
| Low | < $47,833 | 198 |
| Medium | $47,833 – $72,256 | 198 |
| High | > $72,256 | 205 |

### ✅ Actions Taken
| Action | Reason |
|---|---|
| Used percentile-based thresholds (33rd/66th) | Produces balanced segments regardless of income distribution shape |
| Added as categorical feature to model | Provides an interpretable income tier signal rather than a continuous raw value |
| Applied in eligibility rules | Low income + high DTI triggers "Not Eligible" classification |

### 💡 Business Implication
> Income segments allow differentiated loan policies — Low earners may be approved for smaller amounts only, while High earners qualify for the full product range.

---

### 4b — DTI Ratio Capping
```python
df["dti_ratio_capped"] = df["dti_ratio"].clip(upper=100)
```

### ✅ Actions Taken
| Action | Reason |
|---|---|
| Capped `dti_ratio` at 100% | DTI above 100% is financially impossible — values are likely data entry errors |
| Used capped version in all downstream analysis | Prevents extreme outliers from distorting model training and statistical tests |
| Retained original `dti_ratio` column | Preserved for audit trail — allows comparison before and after capping |

### 💡 Business Implication
> Borrowers with DTI above 100% should be automatically flagged for data verification before the application proceeds — the figures likely reflect a data quality issue rather than a real financial position.

---

### 4c — Risk Score
```python
df["credit_score_norm"] = (df["credit_score"] - df["credit_score"].min()) / \
                          (df["credit_score"].max() - df["credit_score"].min())
df["dti_norm"] = df["dti_ratio_capped"] / 100
df["risk_score"] = (df["dti_norm"] * 0.5) + ((1 - df["credit_score_norm"]) * 0.5)
```

### ✅ Actions Taken
| Action | Reason |
|---|---|
| Normalised both inputs to 0–1 scale | Ensures neither credit score nor DTI dominates due to scale differences |
| Applied equal weighting (50/50) | Conservative starting point — weights can be tuned based on business priorities |
| Validated correlation with default (0.33) | Confirms engineered score captures real default signal — not just noise |

### 💡 Business Implication
> The Risk Score provides a single interpretable number (0–1) for loan officers. A score above 0.60 should trigger mandatory secondary review regardless of individual metric values.

---

### 4d — Risk Band
```python
df["risk_band"] = pd.cut(df["risk_score"],
    bins=[-np.inf, 0.35, 0.60, np.inf],
    labels=["Low Risk", "Medium Risk", "High Risk"])
```

### ✅ Actions Taken
| Action | Reason |
|---|---|
| Defined three bands based on risk score distribution | Creates actionable tiers — Low/Medium/High maps directly to approval workflows |
| Used as both a model feature and a reporting dimension | Enables Power BI dashboards to slice default rates by risk tier |

### 💡 Business Implication
> Track the proportion of new applications in each risk band monthly. A rising High Risk share signals deteriorating portfolio quality before defaults materialise — allowing early intervention.

---

### 4e — Loan Eligibility
```python
def eligibility(row):
    if row["credit_score"] >= 650 and row["dti_ratio_capped"] <= 43:
        return "Eligible"
    elif row["credit_score"] >= 580 and row["dti_ratio_capped"] <= 55:
        return "Conditional"
    else:
        return "Not Eligible"
```

### ✅ Actions Taken
| Action | Reason |
|---|---|
| Set Eligible threshold at credit score ≥ 650 and DTI ≤ 43% | Aligns with CFPB and Fannie Mae standard underwriting guidelines |
| Created "Conditional" tier (580–649, DTI ≤ 55%) | Captures borderline applicants who may qualify with additional conditions |
| Used in statistical analysis and modelling | Validates whether engineered eligibility tiers align with actual default outcomes |

### 💡 Business Implication
> The Conditional tier is strategically important — these borrowers are salvageable with the right product design (secured loans, income-based repayment). Blanket rejection of this group leaves revenue on the table.

---

### 4f — Supporting Features
```python
df["is_delinquent"]  = (df["days_delinquent"] > 0).astype(int)
df["loan_to_income"] = (df["loan_amount"] / df["annual_income"]).round(4)
```

### ✅ Actions Taken
| Action | Reason |
|---|---|
| Converted `days_delinquent` to binary `is_delinquent` | Zero-inflated continuous variable not model-friendly — binary flag captures key signal |
| Created `loan_to_income` ratio | Contextualises loan size against repayment capacity |
| Excluded `loan_to_income` from final model | T-test p-value = 0.0787 — not statistically significant at 95% confidence |

---

## Stage 5 — Correlation & Statistical Analysis

### 5a — Correlation with Default

| Feature | Correlation | Direction | Action |
|---|---|---|---|
| `days_delinquent` | **0.88** 🔴 | Higher = more default | **Excluded** — data leakage |
| `risk_score` | 0.33 | Higher = more default | **Included** — engineered feature validated |
| `interest_rate` | 0.20 | Higher = more default | **Included** — captures lender risk pricing |
| `dti_ratio_capped` | 0.20 | Higher = more default | **Included** — statistically significant |
| `credit_score` | **-0.29** 🔵 | Lower = more default | **Included** — strongest legitimate predictor |
| `annual_income` | -0.08 | Lower = slightly more default | **Included** — borderline but financially logical |

### ✅ Actions Taken — Correlations
| Action | Reason |
|---|---|
| Excluded `days_delinquent` from approval model | Correlation of 0.88 = data leakage — information not available at loan approval time |
| Retained `days_delinquent` for monitoring model only | Valid for post-disbursement risk monitoring — not for upfront approval decisions |
| Confirmed `risk_score` correlation of 0.33 | Validates that the engineered composite feature captures real default signal |
| Excluded `age` and `years_employed` | Near-zero correlation — adding noise features hurts model performance |

---

### 5b — Notable Inter-Feature Correlations

| Feature Pair | Correlation | Action |
|---|---|---|
| `credit_score` ↔ `interest_rate` | -0.76 | Both retained — pipeline handles via StandardScaler |
| `credit_score` ↔ `risk_score` | -0.84 | Both retained — each adds unique signal beyond the other |
| `loan_amount` ↔ `monthly_payment` | 0.68 | Both retained — Random Forest handles collinearity naturally |
| `dti_ratio` ↔ `monthly_payment` | 0.64 | Acceptable — different financial concepts despite correlation |
| `annual_income` ↔ `existing_monthly_debt` | 0.62 | Higher earners carry more debt in absolute terms — not a risk concern |

### ✅ Actions Taken — Inter-Feature Correlations
| Action | Reason |
|---|---|
| Retained correlated features for tree-based models | Random Forest and Gradient Boosting handle multicollinearity naturally |
| Applied StandardScaler in pipeline for linear models | Reduces impact of correlated features on Logistic Regression and SVM |
| Did not drop features purely based on inter-correlation | Each feature adds a distinct financial concept even if mathematically correlated |

---

### 5c — T-Test Results (Defaulted vs Non-Defaulted)

| Feature | Default Mean | Non-Default Mean | P-Value | Significant | Action |
|---|---|---|---|---|---|
| `credit_score` | 656.13 | 721.67 | 0.0000 | ✅ YES | Include — 65-point gap is highly significant |
| `dti_ratio_capped` | 56.55 | 46.37 | 0.0000 | ✅ YES | Include — 10-point gap, highly significant |
| `interest_rate` | 11.43 | 10.39 | 0.0000 | ✅ YES | Include — confirms lender pricing reflects risk |
| `annual_income` | $60,854 | $66,405 | 0.0457 | ✅ YES | Include — borderline but statistically valid |
| `days_delinquent` | 101.33 | 3.20 | 0.0000 | ✅ YES | Exclude from approval model — data leakage |
| `risk_score` | 0.58 | 0.43 | 0.0000 | ✅ YES | Include — engineered feature statistically validated |
| `loan_to_income` | 0.41 | 0.37 | 0.0787 | ❌ NO | Exclude from model — difference may be random |

### ✅ Actions Taken — T-Tests
| Action | Reason |
|---|---|
| Used p < 0.05 as significance threshold | Standard 95% confidence level — industry accepted for credit risk analysis |
| Excluded `loan_to_income` from model features | P-value of 0.0787 means difference between defaulters and non-defaulters may be random |
| Confirmed 6 of 7 features are significant | Strong statistical foundation — features are not random noise |
| Used mean differences to inform business rules | Credit score gap (65 pts) and DTI gap (10%) directly informed the eligibility thresholds in Stage 4 |

### 💡 Business Implication
> The $5,551 mean income gap between defaulters ($60,854) and non-defaulters ($66,405) confirms that income matters but is not the dominant factor. A high-income borrower with a poor credit score and high DTI is still high risk — income should supplement, not override, credit and DTI criteria.

---

## Key Findings Summary

### 🔴 High Risk Borrower Profile
A borrower is most likely to default when they:
- Have a **credit score below 650** (default mean: 656)
- Have a **DTI ratio above 55%** (default mean: 56.55%)
- Are a **renter** (31% default rate)
- Work **part-time** (27.7% default rate)
- Take out a **wedding or home improvement loan**
- Are assigned a **high interest rate >12%**

**Recommended actions:**
- Require co-signer or collateral before approval
- Reduce maximum approvable loan amount
- Apply higher interest rate to compensate for risk
- Flag for manual underwriting review

---

### 🟢 Low Risk Borrower Profile
A borrower is least likely to default when they:
- Have a **credit score above 720**
- Have a **DTI ratio below 43%**
- **Own their home** (18.6% default rate)
- Have a **Master's or Associate's degree**
- Take out a **medical expenses loan** (20.6% default rate)
- Have **stable full-time or contract employment**

**Recommended actions:**
- Offer streamlined fast-track approval
- Eligible for preferential interest rates
- Qualify for higher loan amounts
- Minimal additional documentation required

---

### 🟡 Conditional / At-Risk Portfolio Segment
- **Late borrowers (103, 17.1%)** — not yet defaulted but showing early warning signs

**Recommended actions:**
- Proactive outreach before Late converts to Default
- Offer restructuring, payment plans, or temporary grace periods
- Assign a dedicated relationship manager for high-value Late accounts

---

### 📊 Final Feature Selection

| Feature | Type | Action | Reason |
|---|---|---|---|
| `credit_score` | Numerical | ✅ Include | Strongest legitimate predictor |
| `dti_ratio_capped` | Numerical | ✅ Include | Repayment burden — statistically significant |
| `interest_rate` | Numerical | ✅ Include | Lender's embedded risk signal |
| `annual_income` | Numerical | ✅ Include | Repayment capacity |
| `risk_score` | Engineered | ✅ Include | Composite credit + DTI signal validated |
| `loan_amount` | Numerical | ✅ Include | Exposure size |
| `loan_to_income` | Engineered | ✅ Include | Financial logic justifies despite p=0.0787 |
| `monthly_payment` | Numerical | ✅ Include | Cash flow burden |
| `income_segment` | Engineered | ✅ Include | Income tier category |
| `employment_status` | Categorical | ✅ Include | Stability signal |
| `home_ownership` | Categorical | ✅ Include | Asset backing |
| `loan_purpose` | Categorical | ✅ Include | Spending type risk |
| `education_level` | Categorical | ✅ Include | Earning potential proxy |
| `days_delinquent` | Numerical | ❌ Exclude | Data leakage — post-loan information |
| `age` | Numerical | ❌ Exclude | Near-zero correlation with default |
| `state` | Categorical | ❌ Exclude | High cardinality, small sample per state |
| `borrower_id`, `loan_id` | ID | ❌ Exclude | Identifiers — no predictive value |

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| Python (Pandas, NumPy) | Data loading, cleaning, feature engineering |
| Plotly (Express & Graph Objects) | All visualisations |
| Scipy (stats) | T-tests and statistical significance testing |
| Scikit-learn | Pipeline, preprocessing, modelling |
| imbalanced-learn (SMOTE) | Class imbalance correction |
| Power BI | DTI ratio and Risk Score DAX measures |

---

## Next Steps — Modelling

The following models will be trained using a full sklearn Pipeline with:
- `StandardScaler` for numerical features
- `OneHotEncoder` for categorical features
- `SMOTE` for class imbalance correction
- `StratifiedKFold` cross-validation (5 folds)

**Models planned:**
1. Logistic Regression (baseline)
2. Random Forest
3. Gradient Boosting
4. K-Nearest Neighbours
5. Support Vector Machine

**Evaluation metrics:**
- ROC-AUC (primary)
- Recall (priority — missing a defaulter is costly in lending)
- F1-Score
- Confusion Matrix

**Actions after modelling:**
- Select best model based on Recall + AUC balance
- Apply SHAP values for model explainability
- Document feature importances for business stakeholder reporting
- Package model into a scoring pipeline for new applicants

---

*Project by Lesson Shepherd Karidza | Financial Data Analytics Portfolio*
