# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Project: Applied ML Assignment 2 - From Trees to Neural Networks

## Assignment Goal
Compare Gradient Boosted Decision Trees (XGBoost) and Multi-Layer Perceptrons (MLP)
on the Home Credit Default Risk dataset. This is a class assignment — NOT a Kaggle submission.

## Dataset
- Primary file: `application_train.csv`
- Target column: `TARGET` (1 = defaulted, 0 = repaid)
- Mixed data types, missing values, imbalanced classes
- **Do NOT use application_test.csv** — it has no TARGET and is not needed for this assignment
- Supplementary tables (bureau.csv, etc.) are optional — start with application_train.csv only

## Steps to Complete
1. Data preparation & EDA
2. XGBoost model (GBDT)
3. MLP model (scikit-learn MLPClassifier)
4. Side-by-side comparison

## Critical Rules
- NO data leakage: all preprocessing (imputation, scaling, encoding) must be
  fit on training set only, then applied to val/test
- Split application_train.csv into train/val/test (70/15/15) — this is our entire dataset
- MLP requires StandardScaler; tree models do not
- Control random seeds for reproducibility (use 42)

## Required Deliverables
- All visualizations saved as .png files (embedded in PDF report later)
- Training vs validation loss curves for both models
- Feature importance plot (XGBoost)
- Summary comparison table (Accuracy, Precision, Recall, F1, AUC-PR)
- Training time comparison

## Libraries Available
- xgboost, scikit-learn, pandas, numpy, matplotlib, seaborn

## Project Overview

This is a **Home Credit Default Risk** prediction project — a binary classification task predicting whether loan applicants will experience payment difficulties (TARGET: 1=yes, 0=no). It is based on the Kaggle Home Credit competition dataset.

## Environment Setup
```bash
source venv/bin/activate   # activate the local Python 3.14 venv
jupyter lab                # launch Jupyter Lab for notebook development
```

All dependencies (pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, plotly) are pre-installed in `venv/`.

## Data Architecture

All data lives in `home-credit-default-risk/`. The schema is a **star schema** centered on loan applications:
```
application_train.csv   (main table, join key: SK_ID_CURR) ← USE THIS ONLY
    ├── bureau.csv                (previous credits from external bureau, 1-to-many)
    │   └── bureau_balance.csv   (monthly balance history per bureau credit, SK_ID_BUREAU)
    ├── previous_application.csv (past Home Credit loan applications, SK_ID_PREV)
    ├── credit_card_balance.csv  (monthly credit card statements, SK_ID_PREV)
    ├── installments_payments.csv (individual installment payment history, SK_ID_PREV)
    └── POS_CASH_balance.csv     (monthly POS/cash loan balances, SK_ID_PREV)
```

- Training set: 307,511 rows, ~120 features + TARGET
- Column descriptions: `HomeCredit_columns_description.csv` (220+ columns documented)
- Dataset is heavily class-imbalanced (most applicants do NOT default)

## Typical Workflow

1. Load `application_train.csv` as the base
2. Handle missing values (many NaNs are meaningful in credit data)
3. Feature engineering (optional: join supplementary tables with aggregation)
4. Split into train/val/test (70/15/15) — fit all preprocessing on train only
5. Train XGBoost and MLPClassifier; evaluate with Accuracy, F1, AUC-PR

## Loading Data
```python
import pandas as pd
DATA_DIR = 'home-credit-default-risk/'
train = pd.read_csv(DATA_DIR + 'application_train.csv')
```

For large supplementary tables (bureau_balance: ~27M rows, installments: ~13M rows),
consider chunked loading or selective column reads to manage memory.