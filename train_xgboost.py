import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score)

# ── Load data ──────────────────────────────────────────────────────────────────
X_train = pd.read_parquet("X_train.parquet")
X_val   = pd.read_parquet("X_val.parquet")
X_test  = pd.read_parquet("X_test.parquet")
y_train = pd.read_parquet("y_train.parquet").squeeze()
y_val   = pd.read_parquet("y_val.parquet").squeeze()
y_test  = pd.read_parquet("y_test.parquet").squeeze()

SCALE_POS_WEIGHT = 11.4   # ~197880 / 17377
SEED = 42

# ── 1. Learning-rate comparison ────────────────────────────────────────────────
BASE_PARAMS = dict(
    n_estimators        = 2000,
    max_depth           = 6,
    subsample           = 0.8,
    colsample_bytree    = 0.8,
    scale_pos_weight    = SCALE_POS_WEIGHT,
    eval_metric         = "logloss",
    early_stopping_rounds = 50,
    random_state        = SEED,
    tree_method         = "hist",
    device              = "cpu",
    verbosity           = 0,
)

learning_rates = [0.01, 0.1, 0.3]
lr_results = {}

print("=== Learning-rate comparison ===")
for lr in learning_rates:
    t0 = time.time()
    model = XGBClassifier(learning_rate=lr, **BASE_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    elapsed = time.time() - t0
    evals = model.evals_result()
    train_loss = evals["validation_0"]["logloss"]
    val_loss   = evals["validation_1"]["logloss"]
    best_iter  = model.best_iteration
    best_val   = val_loss[best_iter]
    lr_results[lr] = dict(train_loss=train_loss, val_loss=val_loss,
                          best_iter=best_iter, best_val=best_val,
                          time=elapsed)
    print(f"  lr={lr}: best_iter={best_iter}, best_val_logloss={best_val:.4f}, "
          f"time={elapsed:.1f}s")

# Plot all three learning rates on the same graph
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
colors = {"train": "#2196F3", "val": "#F44336"}

for ax, lr in zip(axes, learning_rates):
    r = lr_results[lr]
    iters = range(len(r["train_loss"]))
    ax.plot(iters, r["train_loss"], color=colors["train"], label="Train", linewidth=1.2)
    ax.plot(iters, r["val_loss"],   color=colors["val"],   label="Val",   linewidth=1.2)
    ax.axvline(r["best_iter"], color="gray", linestyle="--", linewidth=0.8,
               label=f"Best iter ({r['best_iter']})")
    ax.set_title(f"Learning rate = {lr}\nbest val logloss = {r['best_val']:.4f}")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Log-loss")
    ax.legend(fontsize=8)

fig.suptitle("XGBoost – Training vs Validation Loss by Learning Rate", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("xgb_learning_rate_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved xgb_learning_rate_comparison.png")

# ── 2. Pick best learning rate ─────────────────────────────────────────────────
best_lr = min(lr_results, key=lambda lr: lr_results[lr]["best_val"])
print(f"\nBest learning rate: {best_lr}")

# ── 3. Hyperparameter tuning on best lr ───────────────────────────────────────
# Grid: n_estimators handled by early stopping; we tune depth, subsample, reg terms
from itertools import product

tuning_grid = {
    "max_depth"  : [4, 6, 8],
    "subsample"  : [0.7, 0.9],
    "reg_alpha"  : [0, 0.1, 1.0],
    "reg_lambda" : [1.0, 5.0],
}

best_score = float("inf")
best_cfg   = {}
print("\n=== Hyperparameter tuning ===")

keys   = list(tuning_grid.keys())
values = list(tuning_grid.values())

for combo in product(*values):
    cfg = dict(zip(keys, combo))
    model = XGBClassifier(
        learning_rate   = best_lr,
        n_estimators    = 1000,
        colsample_bytree= 0.8,
        scale_pos_weight= SCALE_POS_WEIGHT,
        eval_metric     = "logloss",
        early_stopping_rounds = 50,
        random_state    = SEED,
        tree_method     = "hist",
        device          = "cpu",
        verbosity       = 0,
        **cfg,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    val_logloss = model.evals_result()["validation_1"]["logloss"][model.best_iteration]
    if val_logloss < best_score:
        best_score = val_logloss
        best_cfg   = cfg
        print(f"  New best: {cfg} → val_logloss={val_logloss:.4f}")

print(f"\nBest config: {best_cfg}, val_logloss={best_score:.4f}")

# ── 4. Final model training ────────────────────────────────────────────────────
print("\n=== Training final model ===")
t0 = time.time()
final_model = XGBClassifier(
    learning_rate   = best_lr,
    n_estimators    = 1000,
    colsample_bytree= 0.8,
    scale_pos_weight= SCALE_POS_WEIGHT,
    eval_metric     = "logloss",
    early_stopping_rounds = 50,
    random_state    = SEED,
    tree_method     = "hist",
    device          = "cpu",
    verbosity       = 0,
    **best_cfg,
)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=100,
)
train_time = time.time() - t0
print(f"\nFinal model trained in {train_time:.1f}s")
print(f"Best iteration: {final_model.best_iteration}")

# ── 5. Evaluation on validation set ───────────────────────────────────────────
y_pred_proba = final_model.predict_proba(X_val)[:, 1]
y_pred       = (y_pred_proba >= 0.5).astype(int)

acc  = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
rec  = recall_score(y_val, y_pred)
f1   = f1_score(y_val, y_pred)
auc_pr = average_precision_score(y_val, y_pred_proba)

print("\n=== Validation Metrics (final model) ===")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1        : {f1:.4f}")
print(f"  AUC-PR    : {auc_pr:.4f}")
print(f"  Train time: {train_time:.1f}s")

# ── 6. Feature importance plot ─────────────────────────────────────────────────
importances = final_model.feature_importances_
feat_names  = X_train.columns.tolist()
feat_df = pd.DataFrame({"feature": feat_names, "importance": importances})
feat_df = feat_df.sort_values("importance", ascending=False).head(25)

fig, ax = plt.subplots(figsize=(9, 8))
bars = ax.barh(feat_df["feature"][::-1], feat_df["importance"][::-1],
               color="#2196F3", edgecolor="white", linewidth=0.4)
ax.set_xlabel("Feature Importance (gain)", fontsize=11)
ax.set_title("XGBoost – Top 25 Feature Importances\n(Final Tuned Model)", fontsize=12)
ax.tick_params(axis="y", labelsize=8)
plt.tight_layout()
plt.savefig("xgb_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved xgb_feature_importance.png")

# ── 7. Save final model ────────────────────────────────────────────────────────
final_model.save_model("xgb_final_model.json")
print("Saved xgb_final_model.json")

# Summary
print("\n=== Summary ===")
print(f"Best learning rate : {best_lr}")
print(f"Best hyperparams   : {best_cfg}")
print(f"Best iteration     : {final_model.best_iteration}")
print(f"Val log-loss       : {best_score:.4f}")
print(f"Accuracy           : {acc:.4f}")
print(f"Precision          : {prec:.4f}")
print(f"Recall             : {rec:.4f}")
print(f"F1                 : {f1:.4f}")
print(f"AUC-PR             : {auc_pr:.4f}")
print(f"Training time      : {train_time:.1f}s")
