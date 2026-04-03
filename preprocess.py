import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = 'home-credit-default-risk/'
SEED = 42

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR + 'application_train.csv')
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

# ── 1a. HAS_CAR flag from OWN_CAR_AGE (must happen before drop) ──────────────
df['HAS_CAR'] = df['OWN_CAR_AGE'].notna().astype(int)
print(f"\n[1a] HAS_CAR created: {df['HAS_CAR'].sum():,} applicants own a car")

# ── 1b. Drop columns with >50% missing ───────────────────────────────────────
missing_pct = df.isnull().mean()
high_missing = missing_pct[missing_pct > 0.50].index.tolist()
df.drop(columns=high_missing, inplace=True)
print(f"\n[1b] Dropped {len(high_missing)} columns with >50% missing:")
for c in sorted(high_missing):
    print(f"     {c}")
print(f"     → Shape now: {df.shape[0]:,} × {df.shape[1]}")

# ── 2. Fix DAYS_EMPLOYED sentinel (365243 = unemployed) ──────────────────────
sentinel_count = (df['DAYS_EMPLOYED'] == 365243).sum()
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
print(f"\n[2] DAYS_EMPLOYED: replaced {sentinel_count:,} sentinel values (365243) with NaN")

# ── 3. Confirm HAS_CAR ───────────────────────────────────────────────────────
print(f"\n[3] HAS_CAR column present: {df['HAS_CAR'].value_counts().to_dict()}")

# ── 4. Median-impute remaining numeric nulls ──────────────────────────────────
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ('SK_ID_CURR', 'TARGET')]

still_missing = {c: df[c].isnull().sum() for c in num_cols if df[c].isnull().any()}
print(f"\n[4] Median-imputing {len(still_missing)} numeric columns:")
medians = {}
for col, n_miss in still_missing.items():
    med = df[col].median()
    medians[col] = med
    df[col] = df[col].fillna(med)
    print(f"    {col:<45} {n_miss:>6,} nulls → median={med:.4g}")

# ── 5. OCCUPATION_TYPE: fill NaN → 'Unknown' ─────────────────────────────────
occ_missing = df['OCCUPATION_TYPE'].isnull().sum()
df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Unknown')
df = df.copy()  # defragment after many column ops
print(f"\n[5] OCCUPATION_TYPE: {occ_missing:,} NaNs → 'Unknown'")
print(f"    Categories now: {sorted(df['OCCUPATION_TYPE'].unique())}")

# ── 6. One-hot encode categoricals ───────────────────────────────────────────
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols = [c for c in cat_cols if c != 'SK_ID_CURR']
print(f"\n[6] One-hot encoding {len(cat_cols)} categorical columns: {cat_cols}")
df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)
print(f"    → Shape after encoding: {df.shape[0]:,} × {df.shape[1]}")

# ── 7. 70 / 15 / 15 split ────────────────────────────────────────────────────
X = df.drop(columns=['SK_ID_CURR', 'TARGET'])
y = df['TARGET']

# First cut: 70% train, 30% temp  (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
# Second cut: split temp 50/50 → 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

print("\n" + "=" * 60)
print("SPLIT SUMMARY")
print("=" * 60)
for name, X_, y_ in [('Train', X_train, y_train),
                      ('Val  ', X_val,   y_val),
                      ('Test ', X_test,  y_test)]:
    pos = y_.sum()
    pct = pos / len(y_) * 100
    print(f"  {name}: {len(X_):>7,} rows | default rate {pct:.2f}%  ({pos:,} positives)")
print(f"\n  Features: {X_train.shape[1]}")

# ── 8. Save splits ────────────────────────────────────────────────────────────
X_train.to_parquet('X_train.parquet', index=False)
X_val.to_parquet('X_val.parquet',   index=False)
X_test.to_parquet('X_test.parquet', index=False)
y_train.to_frame().to_parquet('y_train.parquet', index=False)
y_val.to_frame().to_parquet('y_val.parquet',     index=False)
y_test.to_frame().to_parquet('y_test.parquet',   index=False)

print("\n[saved] X_train / X_val / X_test / y_train / y_val / y_test  (.parquet)")
print("\nDone. Nothing fitted — medians logged above; re-derive from X_train if needed.")

# ── 9. Quick sanity check: no nulls remain ────────────────────────────────────
remaining_nulls = X_train.isnull().sum().sum()
print(f"\nSanity check — nulls remaining in X_train: {remaining_nulls}")
