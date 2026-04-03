import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score,
                             precision_recall_curve)

# ── Load splits ───────────────────────────────────────────────────────────────
X_train = pd.read_parquet('X_train.parquet')
X_val   = pd.read_parquet('X_val.parquet')
X_test  = pd.read_parquet('X_test.parquet')
y_train = pd.read_parquet('y_train.parquet').squeeze()
y_val   = pd.read_parquet('y_val.parquet').squeeze()
y_test  = pd.read_parquet('y_test.parquet').squeeze()

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

SCALE_POS_WEIGHT = 11.4   # ratio 0:1 from EDA (91.93 / 8.07)
SEED = 42

BASE_PARAMS = dict(
    n_estimators      = 1000,   # high ceiling; early stopping will cut it
    max_depth         = 6,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    scale_pos_weight  = SCALE_POS_WEIGHT,
    eval_metric       = 'aucpr',
    early_stopping_rounds = 50,
    random_state      = SEED,
    n_jobs            = -1,
    verbosity         = 0,
)

# ── Phase 1: Learning rate comparison ────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 1 — Learning Rate Comparison (lr = 0.01, 0.1, 0.3)")
print("=" * 60)

lr_results = {}   # lr -> {model, evals, best_round, val_auprc}

for lr in [0.01, 0.1, 0.3]:
    model = XGBClassifier(learning_rate=lr, **BASE_PARAMS)
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    elapsed = time.time() - t0

    evals       = model.evals_result()
    train_curve = evals['validation_0']['aucpr']
    val_curve   = evals['validation_1']['aucpr']
    best_round  = model.best_iteration
    best_val    = val_curve[best_round]

    lr_results[lr] = dict(
        model=model, train_curve=train_curve,
        val_curve=val_curve, best_round=best_round,
        best_val=best_val, elapsed=elapsed,
    )
    print(f"  lr={lr:<5}  best_round={best_round:>4}  val_auprc={best_val:.4f}"
          f"  time={elapsed:.1f}s")

# ── Plot: learning rate comparison ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {0.01: '#4C72B0', 0.1: '#DD8452', 0.3: '#55A868'}

for lr, res in lr_results.items():
    n = res['best_round'] + 1
    epochs = range(1, len(res['train_curve']) + 1)
    axes[0].plot(epochs, res['train_curve'], color=colors[lr],
                 linestyle='--', alpha=0.5, linewidth=1.2)
    axes[0].plot(epochs, res['val_curve'], color=colors[lr],
                 label=f'lr={lr} (best={res["best_round"]+1})', linewidth=1.8)
    axes[0].axvline(n, color=colors[lr], linestyle=':', alpha=0.7)

    # zoom to first 300 rounds for readability
    zoom_end = min(300, len(res['val_curve']))
    axes[1].plot(range(1, zoom_end+1), res['val_curve'][:zoom_end],
                 color=colors[lr], label=f'lr={lr}', linewidth=1.8)

for ax in axes:
    ax.set_xlabel('Boosting Round')
    ax.set_ylabel('AUC-PR')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes[0].set_title('Full Training — Val AUC-PR by Learning Rate\n(dashed=train, solid=val, dotted=early stop)', fontsize=10)
axes[1].set_title('First 300 Rounds — Val AUC-PR (zoomed)', fontsize=10)
fig.suptitle('XGBoost Learning Rate Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('xgb_learning_rate_comparison.png', dpi=150)
plt.close()
print("\n[saved] xgb_learning_rate_comparison.png")

# ── Pick best LR ─────────────────────────────────────────────────────────────
best_lr = max(lr_results, key=lambda lr: lr_results[lr]['best_val'])
print(f"\nBest learning rate: {best_lr}  (val AUC-PR = {lr_results[best_lr]['best_val']:.4f})")

# ── Phase 2: Hyperparameter tuning ───────────────────────────────────────────
print("\n" + "=" * 60)
print(f"PHASE 2 — Hyperparameter Tuning (lr={best_lr})")
print("=" * 60)

# Grid: vary one dimension at a time from the base
tuning_grid = [
    # (label, overrides)
    ("base",                        {}),
    ("max_depth=4",                 {"max_depth": 4}),
    ("max_depth=8",                 {"max_depth": 8}),
    ("subsample=0.6",               {"subsample": 0.6}),
    ("subsample=1.0",               {"subsample": 1.0}),
    ("reg_alpha=1.0",               {"reg_alpha": 1.0}),
    ("reg_alpha=10",                {"reg_alpha": 10}),
    ("reg_lambda=5",                {"reg_lambda": 5}),
    ("reg_lambda=10",               {"reg_lambda": 10}),
    ("depth4+sub0.6+alpha1",        {"max_depth": 4, "subsample": 0.6, "reg_alpha": 1.0}),
    ("depth8+sub0.8+alpha0.1",      {"max_depth": 8, "subsample": 0.8, "reg_alpha": 0.1}),
]

tuning_results = []
for label, overrides in tuning_grid:
    params = {**BASE_PARAMS, "learning_rate": best_lr, **overrides}
    m = XGBClassifier(**params)
    t0 = time.time()
    m.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_val, y_val)],
          verbose=False)
    elapsed = time.time() - t0
    val_auprc = m.evals_result()['validation_1']['aucpr'][m.best_iteration]
    best_n    = m.best_iteration + 1
    tuning_results.append((label, val_auprc, best_n, elapsed, m, overrides))
    print(f"  {label:<35}  val_auprc={val_auprc:.4f}  rounds={best_n:>4}  t={elapsed:.1f}s")

# Best tuned config
tuning_results.sort(key=lambda x: -x[1])
best_label, best_auprc, best_n, _, final_model, best_overrides = tuning_results[0]
print(f"\nBest config: '{best_label}'  →  val AUC-PR = {best_auprc:.4f}")

# ── Retrain final model cleanly, record time ─────────────────────────────────
print("\n" + "=" * 60)
print("FINAL MODEL — clean retrain with best config")
print("=" * 60)

final_params = {**BASE_PARAMS, "learning_rate": best_lr, **best_overrides}
final_model  = XGBClassifier(**final_params)

t_start = time.time()
final_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False,
)
xgb_train_time = time.time() - t_start

print(f"  Best iteration : {final_model.best_iteration + 1}")
print(f"  Training time  : {xgb_train_time:.2f}s")

# ── Evaluate on validation set ────────────────────────────────────────────────
y_prob_val = final_model.predict_proba(X_val)[:, 1]
# threshold at 0.5 for classification metrics
y_pred_val = (y_prob_val >= 0.5).astype(int)

acc   = accuracy_score(y_val, y_pred_val)
prec  = precision_score(y_val, y_pred_val, zero_division=0)
rec   = recall_score(y_val, y_pred_val, zero_division=0)
f1    = f1_score(y_val, y_pred_val, zero_division=0)
auprc = average_precision_score(y_val, y_prob_val)

print("\n── Validation Metrics ──────────────────────────────────")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1        : {f1:.4f}")
print(f"  AUC-PR    : {auprc:.4f}")
print(f"  Train time: {xgb_train_time:.2f}s")

# ── Feature importance plot ───────────────────────────────────────────────────
importances = pd.Series(
    final_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

top_n = 25
top_imp = importances.head(top_n)

fig, ax = plt.subplots(figsize=(9, 8))
colors_imp = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]
ax.barh(top_imp.index[::-1], top_imp.values[::-1], color=colors_imp[::-1])
ax.set_xlabel('Feature Importance (gain)', fontsize=11)
ax.set_title(f'XGBoost — Top {top_n} Feature Importances\n'
             f'(lr={best_lr}, {best_label})', fontsize=12, fontweight='bold')
ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig('xgb_feature_importance.png', dpi=150)
plt.close()
print("\n[saved] xgb_feature_importance.png")

# ── PR curve plot ─────────────────────────────────────────────────────────────
precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_prob_val)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(recall_vals, precision_vals, color='#4C72B0', linewidth=2,
        label=f'XGBoost (AUC-PR = {auprc:.4f})')
ax.axhline(y_val.mean(), color='gray', linestyle='--', linewidth=1,
           label=f'Baseline (= class prior {y_val.mean():.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve — XGBoost (Validation)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('xgb_pr_curve.png', dpi=150)
plt.close()
print("[saved] xgb_pr_curve.png")

# ── Save results for comparison table later ───────────────────────────────────
results = {
    'model': 'XGBoost',
    'accuracy': acc, 'precision': prec, 'recall': rec,
    'f1': f1, 'auc_pr': auprc,
    'train_time_s': xgb_train_time,
    'best_lr': best_lr,
    'best_config': best_label,
    'best_iteration': final_model.best_iteration + 1,
}
pd.DataFrame([results]).to_csv('xgb_results.csv', index=False)
print("[saved] xgb_results.csv")

# Also save feature importances for reference
importances.to_csv('xgb_feature_importances.csv', header=['importance'])
print("[saved] xgb_feature_importances.csv")

print("\nDone.")
