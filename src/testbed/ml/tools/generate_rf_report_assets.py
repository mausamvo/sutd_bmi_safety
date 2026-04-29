"""Generate report-ready graphs and summary text for RF diagnostics."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(conf_path: Path) -> dict:
    cm_df = pd.read_csv(conf_path, index_col=0)
    cm = cm_df.values
    classes = list(cm_df.index)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("RF Validation Confusion Matrix")

    max_val = cm.max() if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            color = "white" if value > max_val * 0.5 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = OUT_DIR / "rf_confusion_matrix_val.png"
    fig.savefig(out_path)
    plt.close(fig)

    total = cm.sum()
    correct = cm.diagonal().sum()
    val_acc = float(correct / total) if total else 0.0
    return {"val_accuracy": val_acc, "total_windows": int(total)}


def hbar_plot(csv_path: Path, x_col: str, y_col: str, title: str, out_name: str, top_n: int = 12):
    df = pd.read_csv(csv_path).head(top_n).copy()
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.barh(df[y_col], df[x_col], color="#1f77b4")
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name)
    plt.close(fig)


def bar_plot_top_confusions(csv_path: Path):
    df = pd.read_csv(csv_path).copy()
    df["pair"] = df["actual"] + " -> " + df["predicted"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.bar(df["pair"], df["count"], color="#d62728")
    ax.set_xlabel("Confusion pair")
    ax.set_ylabel("Count")
    ax.set_title("Top RF Misclassification Pairs (Validation)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rf_top_confusions_val.png")
    plt.close(fig)


def write_report_summary(metrics: dict):
    confusions = pd.read_csv(ROOT / "rf_top_confusions_val.csv")
    importances = pd.read_csv(ROOT / "rf_feature_importance_val.csv")
    pressure = pd.read_csv(ROOT / "rf_branch_feature_pressure_val.csv")
    bad_leaves = pd.read_csv(ROOT / "rf_bad_leaves_val.csv")

    lines = []
    lines.append("# RF Report Summary")
    lines.append("")
    lines.append("## Core Metrics")
    lines.append(f"- Validation accuracy: {metrics['val_accuracy']:.4f}")
    lines.append(f"- Validation windows: {metrics['total_windows']}")
    lines.append("")
    lines.append("## Top Misclassification Pairs")
    for _, row in confusions.head(5).iterrows():
        lines.append(f"- {row['actual']} -> {row['predicted']}: {int(row['count'])}")
    lines.append("")
    lines.append("## Most Important RF Features")
    for _, row in importances.head(6).iterrows():
        lines.append(f"- {row['feature']}: {row['importance']:.4f}")
    lines.append("")
    lines.append("## Features Dominating Misclassified Paths")
    for _, row in pressure.head(6).iterrows():
        lines.append(f"- {row['feature']}: {int(row['path_hits_on_misclassified'])} path hits")
    lines.append("")
    lines.append("## High-Error Branches (Leaves)")
    for _, row in bad_leaves.head(5).iterrows():
        lines.append(
            "- tree "
            f"{int(row['tree'])}, leaf {int(row['leaf'])}, support {int(row['support'])}, "
            f"misclassified {int(row['misclassified'])}, error_rate {row['error_rate']:.2f}"
        )
    lines.append("")
    lines.append("## Figure Captions")
    lines.append("- rf_confusion_matrix_val.png: Validation confusion matrix for RF classifier.")
    lines.append("- rf_top_confusions_val.png: Highest-frequency class confusions.")
    lines.append("- rf_feature_importance_val.png: Top feature importances from RF.")
    lines.append("- rf_branch_feature_pressure_val.png: Features most often traversed on misclassified paths.")

    (OUT_DIR / "rf_report_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    metrics = plot_confusion_matrix(ROOT / "confusion_matrix_rf_val.csv")

    bar_plot_top_confusions(ROOT / "rf_top_confusions_val.csv")

    hbar_plot(
        ROOT / "rf_feature_importance_val.csv",
        x_col="importance",
        y_col="feature",
        title="Top RF Feature Importances (Validation)",
        out_name="rf_feature_importance_val.png",
        top_n=12,
    )

    hbar_plot(
        ROOT / "rf_branch_feature_pressure_val.csv",
        x_col="path_hits_on_misclassified",
        y_col="feature",
        title="Feature Pressure on Misclassified Paths (Validation)",
        out_name="rf_branch_feature_pressure_val.png",
        top_n=12,
    )

    write_report_summary(metrics)
    print(f"Saved report assets to: {OUT_DIR}")


if __name__ == "__main__":
    main()
