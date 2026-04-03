import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score)

# ── Load data ──────────────────────────────────────────────────────────────────
X_train = pd.read_parquet("X_train.parquet")
X_val   = pd.read_parquet("X_val.parquet")
X_test  = pd.read_parquet("X_test.parquet")
y_train = pd.read_parquet("y_train.parquet").squeeze()
y_val   = pd.read_parquet("y_val.parquet").squeeze()
y_test  = pd.read_parquet("y_test.parquet").squeeze()

# ── Scale — fit on train only ──────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

SEED = 42

def eval_metrics(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    return dict(
        accuracy  = accuracy_score(y, pred),
        precision = precision_score(y, pred, zero_division=0),
        recall    = recall_score(y, pred),
        f1        = f1_score(y, pred, zero_division=0),
        auc_pr    = average_precision_score(y, proba),
    )

BASE_MLP = dict(
    max_iter            = 200,
    early_stopping      = True,
    validation_fraction = 0.1,
    n_iter_no_change    = 15,
    random_state        = SEED,
    class_weight        = "balanced",   # handled via sample_weight below
    # Note: MLPClassifier has no class_weight param; use sample_weight in fit
)

# MLPClassifier doesn't accept class_weight — compute sample weights manually
class_counts = y_train.value_counts()
n_total = len(y_train)
n_classes = len(class_counts)
class_weight_map = {c: n_total / (n_classes * cnt) for c, cnt in class_counts.items()}
sample_weights = y_train.map(class_weight_map).values

BASE_FIT = dict(sample_weight=sample_weights)

BASE_PARAMS = dict(
    activation          = "relu",
    learning_rate_init  = 0.001,
    solver              = "adam",
    max_iter            = 200,
    early_stopping      = True,
    validation_fraction = 0.1,
    n_iter_no_change    = 15,
    random_state        = SEED,
)

# ── 1. Architecture comparison ─────────────────────────────────────────────────
architectures = [(64,), (128, 64), (256, 128, 64)]
arch_results = {}

print("=== Architecture comparison ===")
for arch in architectures:
    t0 = time.time()
    model = MLPClassifier(hidden_layer_sizes=arch, **BASE_PARAMS)
    model.fit(X_train_s, y_train, **BASE_FIT)
    elapsed = time.time() - t0
    m = eval_metrics(model, X_val_s, y_val)
    arch_results[arch] = dict(**m, time=elapsed, model=model)
    print(f"  arch={arch}: F1={m['f1']:.4f}, AUC-PR={m['auc_pr']:.4f}, "
          f"iters={model.n_iter_}, time={elapsed:.1f}s")

# Plot AUC-PR and F1 per architecture
arch_labels = [str(a) for a in architectures]
auc_pr_vals = [arch_results[a]["auc_pr"] for a in architectures]
f1_vals     = [arch_results[a]["f1"] for a in architectures]

x = np.arange(len(architectures))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, auc_pr_vals, width, label="AUC-PR", color="#2196F3")
bars2 = ax.bar(x + width/2, f1_vals,     width, label="F1",     color="#4CAF50")

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(arch_labels, fontsize=10)
ax.set_xlabel("Hidden Layer Architecture")
ax.set_ylabel("Score")
ax.set_title("MLP – AUC-PR and F1 by Architecture\n(relu, lr=0.001, class-weighted)")
ax.legend()
ax.set_ylim(0, max(max(auc_pr_vals), max(f1_vals)) * 1.2)
plt.tight_layout()
plt.savefig("mlp_architecture_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved mlp_architecture_comparison.png")

# Pick best architecture by AUC-PR
best_arch = max(arch_results, key=lambda a: arch_results[a]["auc_pr"])
print(f"\nBest architecture: {best_arch} (AUC-PR={arch_results[best_arch]['auc_pr']:.4f})")

# ── 2. Activation comparison ───────────────────────────────────────────────────
print("\n=== Activation comparison ===")
act_results = {}
for act in ["relu", "tanh"]:
    model = MLPClassifier(hidden_layer_sizes=best_arch, activation=act,
                          **{k: v for k, v in BASE_PARAMS.items() if k != "activation"})
    model.fit(X_train_s, y_train, **BASE_FIT)
    m = eval_metrics(model, X_val_s, y_val)
    act_results[act] = dict(**m, model=model)
    print(f"  activation={act}: F1={m['f1']:.4f}, AUC-PR={m['auc_pr']:.4f}")

best_act = max(act_results, key=lambda a: act_results[a]["auc_pr"])
print(f"Best activation: {best_act}")

# ── 3. Learning rate comparison ────────────────────────────────────────────────
print("\n=== Learning rate comparison ===")
lr_results = {}
for lr in [0.001, 0.01, 0.1]:
    model = MLPClassifier(hidden_layer_sizes=best_arch, activation=best_act,
                          learning_rate_init=lr,
                          **{k: v for k, v in BASE_PARAMS.items()
                             if k not in ("activation", "learning_rate_init")})
    model.fit(X_train_s, y_train, **BASE_FIT)
    m = eval_metrics(model, X_val_s, y_val)
    lr_results[lr] = dict(**m, model=model)
    print(f"  lr={lr}: F1={m['f1']:.4f}, AUC-PR={m['auc_pr']:.4f}, iters={model.n_iter_}")

best_lr = max(lr_results, key=lambda lr: lr_results[lr]["auc_pr"])
print(f"Best learning rate: {best_lr}")

# ── 4. Final model ─────────────────────────────────────────────────────────────
print("\n=== Training final model ===")
t0 = time.time()
final_model = MLPClassifier(
    hidden_layer_sizes  = best_arch,
    activation          = best_act,
    learning_rate_init  = best_lr,
    solver              = "adam",
    max_iter            = 500,
    early_stopping      = True,
    validation_fraction = 0.1,
    n_iter_no_change    = 20,
    random_state        = SEED,
)
final_model.fit(X_train_s, y_train, **BASE_FIT)
train_time = time.time() - t0

print(f"Converged after {final_model.n_iter_} iterations in {train_time:.1f}s")

# ── 5. Evaluation ──────────────────────────────────────────────────────────────
metrics = eval_metrics(final_model, X_val_s, y_val)
print("\n=== Validation Metrics (final model) ===")
print(f"  Accuracy  : {metrics['accuracy']:.4f}")
print(f"  Precision : {metrics['precision']:.4f}")
print(f"  Recall    : {metrics['recall']:.4f}")
print(f"  F1        : {metrics['f1']:.4f}")
print(f"  AUC-PR    : {metrics['auc_pr']:.4f}")
print(f"  Train time: {train_time:.1f}s")

# ── 6. Loss curve ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(final_model.loss_curve_, color="#2196F3", linewidth=1.5, label="Training loss")
if hasattr(final_model, "validation_scores_"):
    # validation_scores_ tracks the metric used for early stopping (default: loss)
    pass
if hasattr(final_model, "best_validation_score_"):
    ax.axhline(final_model.best_validation_score_, color="gray", linestyle="--",
               linewidth=0.8, label=f"Best val score ({final_model.best_validation_score_:.4f})")

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title(f"MLP – Training Loss Curve\narch={best_arch}, activation={best_act}, lr={best_lr}")
ax.legend()
plt.tight_layout()
plt.savefig("mlp_loss_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved mlp_loss_curve.png")

# ── 7. Summary ─────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"Best architecture  : {best_arch}")
print(f"Best activation    : {best_act}")
print(f"Best learning rate : {best_lr}")
print(f"Iterations         : {final_model.n_iter_}")
print(f"Accuracy           : {metrics['accuracy']:.4f}")
print(f"Precision          : {metrics['precision']:.4f}")
print(f"Recall             : {metrics['recall']:.4f}")
print(f"F1                 : {metrics['f1']:.4f}")
print(f"AUC-PR             : {metrics['auc_pr']:.4f}")
print(f"Training time      : {train_time:.1f}s")
