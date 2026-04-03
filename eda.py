import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'home-credit-default-risk/'
df = pd.read_csv(DATA_DIR + 'application_train.csv')

# ── 1. Shape & dtypes ────────────────────────────────────────────────────────
print("=" * 60)
print("SHAPE")
print("=" * 60)
print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")

print("\n" + "=" * 60)
print("DTYPE SUMMARY")
print("=" * 60)
dtype_counts = df.dtypes.value_counts()
for dt, cnt in dtype_counts.items():
    print(f"  {str(dt):<12} {cnt} columns")

# ── 2. Missing values ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MISSING VALUES  (columns with any missing, sorted desc)")
print("=" * 60)
missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing = missing[missing > 0]
print(f"  Columns with missing data: {len(missing)} / {df.shape[1]}")
print()
for col, pct in missing.items():
    bar = '█' * int(pct / 2)
    print(f"  {col:<45} {pct:5.1f}%  {bar}")

# ── 3. Class balance ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TARGET CLASS BALANCE")
print("=" * 60)
vc = df['TARGET'].value_counts()
for v, cnt in vc.items():
    label = "Repaid (0)" if v == 0 else "Default (1)"
    pct = cnt / len(df) * 100
    print(f"  {label}: {cnt:>7,}  ({pct:.2f}%)")
imbalance_ratio = vc[0] / vc[1]
print(f"\n  Imbalance ratio (0:1) = {imbalance_ratio:.1f}:1")

# ── 4. Top numeric features (by variance, excluding ID/target) ───────────────
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ('SK_ID_CURR', 'TARGET')]
print("\n" + "=" * 60)
print(f"NUMERIC FEATURES: {len(num_cols)} columns")
cat_cols = df.select_dtypes(include='object').columns.tolist()
print(f"CATEGORICAL FEATURES: {len(cat_cols)} columns")
print("=" * 60)

# Describe top 10 by variance (normalised)
variances = df[num_cols].var(ddof=0)
top10 = variances.nlargest(10).index.tolist()
print("\nTop 10 numeric features by variance:")
desc = df[top10].describe().T[['mean', 'std', 'min', '50%', 'max']]
desc.columns = ['mean', 'std', 'min', 'median', 'max']
print(desc.to_string())

# ── PLOTS ─────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')

# Plot 1: Class balance bar chart
fig, ax = plt.subplots(figsize=(5, 4))
colors = ['#4C72B0', '#DD8452']
bars = ax.bar(['Repaid (0)', 'Default (1)'], vc.values, color=colors, edgecolor='white', linewidth=1.2)
for bar, cnt in zip(bars, vc.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
            f'{cnt:,}\n({cnt/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=10)
ax.set_title('TARGET Class Balance', fontsize=13, fontweight='bold')
ax.set_ylabel('Count')
ax.set_ylim(0, vc.values.max() * 1.15)
plt.tight_layout()
plt.savefig('eda_class_balance.png', dpi=150)
plt.close()
print("\n[saved] eda_class_balance.png")

# Plot 2: Missing value heatmap (top 40 most-missing columns)
top_missing = missing.head(40)
fig, ax = plt.subplots(figsize=(10, 8))
colors_bar = ['#d62728' if p > 50 else '#ff7f0e' if p > 20 else '#aec7e8' for p in top_missing.values]
ax.barh(top_missing.index[::-1], top_missing.values[::-1], color=colors_bar[::-1])
ax.axvline(50, color='red', linestyle='--', alpha=0.6, label='>50% missing')
ax.axvline(20, color='orange', linestyle='--', alpha=0.6, label='>20% missing')
ax.set_xlabel('Missing %')
ax.set_title('Missing Value % — Top 40 Columns', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('eda_missing_values.png', dpi=150)
plt.close()
print("[saved] eda_missing_values.png")

# Plot 3: Distributions of 9 interpretable numeric features vs TARGET
interpret_cols = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'DAYS_BIRTH', 'DAYS_EMPLOYED', 'EXT_SOURCE_1',
    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_GOODS_PRICE',
]
# keep only columns that actually exist
interpret_cols = [c for c in interpret_cols if c in df.columns]

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()
for i, col in enumerate(interpret_cols):
    ax = axes[i]
    for target_val, label, color in [(0, 'Repaid', '#4C72B0'), (1, 'Default', '#DD8452')]:
        vals = df.loc[df['TARGET'] == target_val, col].dropna()
        # clip extreme outliers to 99th pct for readability
        p99 = vals.quantile(0.99)
        vals = vals.clip(upper=p99)
        ax.hist(vals, bins=50, alpha=0.6, label=label, color=color, density=True)
    ax.set_title(col, fontsize=9, fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=8)
# hide unused
for j in range(len(interpret_cols), len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Feature Distributions by TARGET (density, clipped at 99th pct)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_feature_distributions.png', dpi=150)
plt.close()
print("[saved] eda_feature_distributions.png")

# Plot 4: Correlation of numeric features with TARGET (top 20 abs corr)
corr_with_target = df[num_cols + ['TARGET']].corr()['TARGET'].drop('TARGET')
top_corr = corr_with_target.abs().nlargest(20).index
corr_vals = corr_with_target[top_corr].sort_values()

fig, ax = plt.subplots(figsize=(8, 7))
colors_corr = ['#DD8452' if v > 0 else '#4C72B0' for v in corr_vals.values]
ax.barh(corr_vals.index, corr_vals.values, color=colors_corr)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Top 20 Numeric Features — Pearson Correlation with TARGET', fontsize=11, fontweight='bold')
ax.set_xlabel('Correlation')
plt.tight_layout()
plt.savefig('eda_target_correlation.png', dpi=150)
plt.close()
print("[saved] eda_target_correlation.png")

print("\nDone.")
