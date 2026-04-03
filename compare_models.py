import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Results from training runs ─────────────────────────────────────────────────
results = {
    "XGBoost": dict(
        accuracy  = 0.8915,
        precision = 0.2650,
        recall    = 0.1941,
        f1        = 0.2241,
        auc_pr    = 0.1868,
        train_time= 7.1,
    ),
    "MLP": dict(
        accuracy  = 0.6682,
        precision = 0.1523,
        recall    = 0.6807,
        f1        = 0.2488,
        auc_pr    = 0.2210,
        train_time= 23.4,
    ),
}

metrics      = ["accuracy", "precision", "recall", "f1", "auc_pr"]
metric_labels= ["Accuracy", "Precision", "Recall", "F1", "AUC-PR"]
models       = list(results.keys())
colors       = {"XGBoost": "#2196F3", "MLP": "#FF9800"}

xgb_vals = [results["XGBoost"][m] for m in metrics]
mlp_vals = [results["MLP"][m]     for m in metrics]

x     = np.arange(len(metrics))
width = 0.35

fig = plt.figure(figsize=(12, 8))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel 1: grouped bar chart of all metrics ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])   # full top row
bars1 = ax1.bar(x - width/2, xgb_vals, width, label="XGBoost", color=colors["XGBoost"])
bars2 = ax1.bar(x + width/2, mlp_vals, width, label="MLP",     color=colors["MLP"])

for bar in list(bars1) + list(bars2):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

ax1.set_xticks(x)
ax1.set_xticklabels(metric_labels, fontsize=11)
ax1.set_ylabel("Score")
ax1.set_ylim(0, 1.15)
ax1.set_title("XGBoost vs MLP — Validation Metrics", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.axhline(0, color="black", linewidth=0.5)

# ── Panel 2: radar / spider chart ─────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0], polar=True)
N = len(metrics)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]   # close polygon

for model, vals_list, color in [("XGBoost", xgb_vals, colors["XGBoost"]),
                                 ("MLP",     mlp_vals, colors["MLP"])]:
    vals = vals_list + vals_list[:1]
    ax2.plot(angles, vals, color=color, linewidth=2, label=model)
    ax2.fill(angles, vals, color=color, alpha=0.15)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(metric_labels, fontsize=9)
ax2.set_ylim(0, 1)
ax2.set_title("Radar Chart", fontsize=11, pad=15)
ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

# ── Panel 3: training time bar ─────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
times  = [results[m]["train_time"] for m in models]
bar_colors = [colors[m] for m in models]
bars = ax3.bar(models, times, color=bar_colors, width=0.4)
for bar, t in zip(bars, times):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{t:.1f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax3.set_ylabel("Seconds")
ax3.set_title("Training Time (Val set, final model)", fontsize=11)
ax3.set_ylim(0, max(times) * 1.25)

plt.suptitle("Model Comparison: XGBoost vs MLP\n(Home Credit Default Risk — Validation Set)",
             fontsize=14, fontweight="bold", y=1.01)

plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved model_comparison.png")
