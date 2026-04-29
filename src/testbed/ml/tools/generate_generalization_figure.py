"""Generate side-by-side generalization comparison figure."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report_assets"

# Load data
df = pd.read_csv(OUT_DIR / "rf_generalization_comparison.csv")

fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

presets = df["preset"].values
x = np.arange(len(presets))
width = 0.35

within = df["baseline validation set"].astype(float).values
cross = df["unseen dataset"].astype(float).values

bars1 = ax.bar(x - width/2, within, width, label="Baseline", color="#2ca02c", alpha=0.8)
bars2 = ax.bar(x + width/2, cross, width, label="Unseen", color="#d62728", alpha=0.8)

ax.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
ax.set_title("Accuracy of Different Feature Presets", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(presets, fontsize=10)
ax.legend(fontsize=10, loc="lower left")
ax.set_ylim([0.75, 0.85])
ax.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

fig.tight_layout()
fig.savefig(OUT_DIR / "rf_generalization_comparison.png", dpi=200)
plt.close(fig)

print(f"Saved: {OUT_DIR}/rf_generalization_comparison.png")
