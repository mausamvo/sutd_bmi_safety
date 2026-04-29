"""Generate baseline-only report figures."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report_assets"

# Confusion matrix heatmap
cm_df = pd.read_csv(ROOT / "confusion_matrix_rf_val.csv", index_col=0)
cm = cm_df.values
classes = list(cm_df.index)

fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(classes, fontsize=9)
ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
ax.set_ylabel("Actual Label", fontsize=10, fontweight="bold")
ax.set_title("Baseline Confusion Matrix (Validation)", fontsize=11, fontweight="bold")

max_val = cm.max() if cm.size else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        value = int(cm[i, j])
        color = "white" if value > max_val * 0.5 else "black"
        ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=8)

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(OUT_DIR / "baseline_confusion_matrix.png")
plt.close(fig)

# Feature importance horizontal bar
imp_df = pd.read_csv(ROOT / "rf_feature_importance_val.csv").head(12)
imp_df = imp_df.iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
ax.barh(imp_df["feature"], imp_df["importance"], color="#1f77b4", alpha=0.8)
ax.set_xlabel("Importance Score", fontsize=10, fontweight="bold")
ax.set_title("Baseline Feature Importances", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "baseline_feature_importance.png")
plt.close(fig)

# Top confusion pairs bar chart
conf_df = pd.read_csv(ROOT / "rf_top_confusions_val.csv")
conf_df["pair"] = conf_df["actual"] + " → " + conf_df["predicted"]

fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
ax.bar(conf_df["pair"], conf_df["count"], color="#d62728", alpha=0.8)
ax.set_ylabel("Count", fontsize=10, fontweight="bold")
ax.set_title("Top Misclassification Pairs (Baseline)", fontsize=11, fontweight="bold")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "baseline_top_confusions.png")
plt.close(fig)

# Feature pressure on misclassified paths
pressure_df = pd.read_csv(ROOT / "rf_branch_feature_pressure_val.csv").head(12)
pressure_df = pressure_df.iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
ax.barh(pressure_df["feature"], pressure_df["path_hits_on_misclassified"], color="#ff7f0e", alpha=0.8)
ax.set_xlabel("Path Hits on Misclassified Samples", fontsize=10, fontweight="bold")
ax.set_title("Feature Pressure on Error Paths (Baseline)", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "baseline_feature_pressure.png")
plt.close(fig)

print("✓ Baseline confusion matrix saved")
print("✓ Baseline feature importance saved")
print("✓ Baseline top confusions saved")
print("✓ Baseline feature pressure saved")
