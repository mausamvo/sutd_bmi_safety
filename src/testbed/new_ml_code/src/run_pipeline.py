"""
End-to-end pipeline for one subject (config.SUBJECT, dataset_cleaned):
  1. Build file-level manifests: train/val (from the 80% baseline train_pool),
     test (the untouched 20% baseline split -- the headline "baseline
     accuracy"), and one split per attack condition (config.ATTACK_CONDITIONS).
  2. Window + compute CWT & TD features for each split (cached).
  3. Search classical (non-NN) models on the TD features -- RandomForest,
     ExtraTrees, HistGradientBoosting, each hyperparameter-tuned by grouped
     (by source file) cross-validation on the train split only -- and keep
     the best one (see src/classical.py).
  4. Train EMGNetFusion on train, model-select on val -- repeated across
     config.TORCH_SEEDS (same data split every time, different init/shuffling).
  5. Select the ensemble's NN/classical blend weight by trying every
     candidate in config.ENSEMBLE_WEIGHT_CANDIDATES and keeping whichever
     scores best -- evaluated ONLY on the internal val split, averaged
     across seeds, never on test/attack. This mirrors the leakage-safe
     discipline used for the file-level split and the classical-model
     search: the weight is picked before it is ever compared against the
     data it will later be judged on.
  6. For each seed: evaluate EMGNetFusion alone, the winning classical model
     alone (fixed across seeds -- it doesn't depend on the NN's init), and
     their soft-voting ensemble at the selected weight (the headline
     reported model) on val/test and every attack split.
  7. Write outputs/metrics.json, outputs/run_log.md, outputs/confusion_matrices.png,
     outputs/training_curve.png, outputs/emgnet_s1.pt (first seed's NN weights).

Run: python3 -m src.run_pipeline   (from the new_ml_code/ directory)
"""
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from . import config
from .classical import fit_best_classical_model
from .data import build_manifests
from .dataset import build_cwt_split, build_td_split
from .evaluate import accuracy_from_proba, evaluate_split, predict_proba_nn
from .model import EMGNetFusion, count_parameters, paper_param_count_check
from .train import train_emgnet

EVAL_SPLITS = ["val", "test"] + config.ATTACK_CONDITIONS
DISPLAY_NAME = {
    "val": "Baseline (internal val, model selection only)",
    "test": "Baseline accuracy (20% held-out test)",
    "attack_50hz": "Attack: 50 Hz interference",
    "attack_laser": "Attack: laser",
    "attack_5g": "Attack: 5G (incomplete session)",
    "attack_5g_5cm": "Attack: 5G at 5 cm",
    "attack_5g_5cm_runB": "Attack: 5G at 5 cm, run B (suspect)",
}
COMPONENTS = ["nn", "rf", "ensemble"]
# COMPONENT_NAME["rf"] is filled in once the classical-model search (Step 3)
# picks a winner -- it isn't necessarily a RandomForest any more, but the
# internal key stays "rf" for "the classical/non-neural-net component"
# throughout the code, to avoid a much larger rename.
COMPONENT_NAME = {"nn": "EMGNetFusion (CNN+TD)", "rf": "classical model (TD only)",
                  "ensemble": "Ensemble (headline model)"}


def main():
    t0 = time.time()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Step 0: sanity-check the EMGNet reconstruction against the paper")
    print("=" * 70)
    paper_param_count_check()

    print("\n" + "=" * 70)
    print("Step 1: build file-level manifests (80/20 baseline split)")
    print("=" * 70)
    manifests = build_manifests()
    for name, mdf in manifests.items():
        print(f"  {name:14s} {len(mdf):4d} files")

    print("\n" + "=" * 70)
    print("Step 2: compute TD features per split (cached to outputs/cache/) -- "
          "CWT scalograms are deliberately NOT loaded yet, see Step 3")
    print("=" * 70)
    td_data = {}
    for split in ["train"] + EVAL_SPLITS:
        X_td, y, window_manifest = build_td_split(manifests[split], split)
        td_data[split] = (X_td, y, window_manifest)
        print(f"  {split:14s} X_td={X_td.shape} y={y.shape}")

    X_td_train, y_train, wm_train = td_data["train"]

    print("\n" + "=" * 70)
    print("Step 3: search classical (non-NN) models on TD features -- grouped "
          "CV by source FILE on the train split, so no CV fold ever splits one "
          "file's windows across folds (same discipline as the outer split). "
          "Deliberately done BEFORE loading the much larger CWT arrays: doing "
          "this parallel, multi-process search with ~2.7GB of CWT tensors also "
          "resident in memory was observed to crash the process.")
    print("=" * 70)
    best_classical_name, rf, classical_search_results = fit_best_classical_model(
        X_td_train, y_train, groups_train=wm_train["path"].values)
    COMPONENT_NAME["rf"] = f"{best_classical_name} (TD only)"
    rf_proba = {split: rf.predict_proba(td_data[split][0]) for split in EVAL_SPLITS}
    for split in EVAL_SPLITS:
        res = accuracy_from_proba(rf_proba[split], td_data[split][1], td_data[split][2])
        print(f"  {best_classical_name}  {DISPLAY_NAME[split]:44s} window_acc={res['overall_accuracy']:.4f}  "
              f"file_majority_acc={res['file_majority_accuracy']:.4f}")

    print("\n" + "=" * 70)
    print("Step 3b: now load CWT scalograms per split (cached to outputs/cache/)")
    print("=" * 70)
    data = {}
    for split in ["train"] + EVAL_SPLITS:
        X_cwt, y, window_manifest = build_cwt_split(manifests[split], split)
        X_td, y_td, _ = td_data[split]
        assert (y == y_td).all(), f"CWT/TD label mismatch for split '{split}'"
        data[split] = (X_cwt, X_td, y, window_manifest)
        print(f"  {split:14s} X_cwt={X_cwt.shape}")

    X_cwt_train = data["train"][0]
    X_cwt_val, X_td_val, y_val, _ = data["val"]

    print("\n" + "=" * 70)
    print(f"Step 4: train EMGNetFusion across {len(config.TORCH_SEEDS)} seeds "
          f"{config.TORCH_SEEDS} (ensemble weight not applied yet -- see Step 4b)")
    print("=" * 70)
    per_seed_nn = []  # {seed, history, nn_res: {split: res}, nn_proba: {split: array}}
    first_model = None
    for seed in config.TORCH_SEEDS:
        print(f"\n--- seed {seed} ---")
        model, history, td_mean, td_std = train_emgnet(
            X_cwt_train, X_td_train, y_train, X_cwt_val, X_td_val, y_val, seed=seed)
        if first_model is None:
            first_model = model

        nn_res_by_split, nn_proba_by_split = {}, {}
        for split in EVAL_SPLITS:
            X_cwt, X_td, y, window_manifest = data[split]
            nn_res_by_split[split] = evaluate_split(model, X_cwt, X_td, y, td_mean, td_std, window_manifest)
            nn_proba_by_split[split] = predict_proba_nn(model, X_cwt, X_td, td_mean, td_std)
            print(f"  seed {seed}  {DISPLAY_NAME[split]:44s}  "
                  f"nn_acc={nn_res_by_split[split]['overall_accuracy']:.3f}  "
                  f"(file-majority: nn={nn_res_by_split[split]['file_majority_accuracy']:.3f})")
        per_seed_nn.append({"seed": seed, "history": history,
                             "nn_res": nn_res_by_split, "nn_proba": nn_proba_by_split})

    print("\n" + "=" * 70)
    print("Step 4b: select the ensemble blend weight -- try every candidate in "
          "config.ENSEMBLE_WEIGHT_CANDIDATES, scored ONLY on the internal val "
          "split (averaged across seeds), never on test/attack. Keep the winner.")
    print("=" * 70)
    weight_search_results = []
    for w in config.ENSEMBLE_WEIGHT_CANDIDATES:
        val_accs = []
        for sr in per_seed_nn:
            ens_proba = w * sr["nn_proba"]["val"] + (1 - w) * rf_proba["val"]
            res = accuracy_from_proba(ens_proba, data["val"][2], data["val"][3])
            val_accs.append(res["overall_accuracy"])
        mean_acc = float(np.mean(val_accs))
        weight_search_results.append({"weight": w, "val_window_acc_mean": mean_acc,
                                       "val_window_acc_per_seed": val_accs})
        print(f"  weight={w:.2f} (NN)  ->  val window accuracy = {mean_acc:.4f} "
              f"(per seed: {', '.join(f'{a:.3f}' for a in val_accs)})")

    best_candidate = max(weight_search_results, key=lambda r: r["val_window_acc_mean"])
    selected_weight = best_candidate["weight"]
    print(f"  -> selected NN weight = {selected_weight:.2f} "
          f"(val window accuracy {best_candidate['val_window_acc_mean']:.4f})")

    print("\n" + "=" * 70)
    print(f"Step 4c: apply the selected weight ({selected_weight:.2f}) to compute "
          f"final ensemble results on every split")
    print("=" * 70)
    per_seed_results = []  # {seed, history, results: {component: {split: res}}}
    for sr in per_seed_nn:
        seed, history = sr["seed"], sr["history"]
        seed_results = {c: {} for c in COMPONENTS}
        for split in EVAL_SPLITS:
            X_cwt, X_td, y, window_manifest = data[split]
            nn_res = sr["nn_res"][split]
            rf_res = accuracy_from_proba(rf_proba[split], y, window_manifest)
            ens_proba = selected_weight * sr["nn_proba"][split] + (1 - selected_weight) * rf_proba[split]
            ens_res = accuracy_from_proba(ens_proba, y, window_manifest)

            seed_results["nn"][split] = nn_res
            seed_results["rf"][split] = rf_res
            seed_results["ensemble"][split] = ens_res
            print(f"  seed {seed}  {DISPLAY_NAME[split]:44s}  "
                  f"nn_acc={nn_res['overall_accuracy']:.3f}  rf_acc={rf_res['overall_accuracy']:.3f}  "
                  f"ensemble_acc={ens_res['overall_accuracy']:.3f} "
                  f"(file-majority: nn={nn_res['file_majority_accuracy']:.3f} "
                  f"rf={rf_res['file_majority_accuracy']:.3f} ens={ens_res['file_majority_accuracy']:.3f})")

        # Baseline test accuracy restricted to sessions recorded on the SAME
        # setup as the attacks. Baseline-vs-attack is only a controlled
        # comparison within that group; the overall test number mixes setups
        # for subjects whose baselines span both (e.g. S2).
        matched = _matched_subset_result(sr, rf_proba, data, selected_weight)
        if matched is not None:
            seed_results["ensemble"]["test_matched"] = matched
            print(f"  seed {seed}  {'Baseline, attack-matched setup only':44s}  "
                  f"ensemble_acc={matched['overall_accuracy']:.3f} "
                  f"(file-majority: {matched['file_majority_accuracy']:.3f}, "
                  f"{matched['n_files']} files)")
        per_seed_results.append({"seed": seed, "history": history, "results": seed_results})

    print("\n" + "=" * 70)
    print("Aggregate across seeds (mean +/- std) -- headline = ensemble")
    print("=" * 70)
    aggregate = _aggregate_seeds(per_seed_results)
    for split in EVAL_SPLITS:
        for comp in COMPONENTS:
            a = aggregate[comp][split]
            print(f"  [{comp:8s}] {DISPLAY_NAME[split]:44s} "
                  f"window_acc={a['mean']:.4f}+/-{a['std']:.4f}  "
                  f"file_majority_acc={a['majority_mean']:.4f}+/-{a['majority_std']:.4f}")

    print("\n" + "=" * 70)
    print("Step 5: write outputs")
    print("=" * 70)
    torch.save(first_model.state_dict(), config.OUTPUT_DIR / f"emgnet_{config.SUBJECT.lower()}.pt")

    n_params = count_parameters(first_model)
    with open(config.OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump({
            "model_parameters_nn": n_params,
            "best_classical_model": best_classical_name,
            "classical_model_search": classical_search_results,
            "ensemble_weight_search": weight_search_results,
            "selected_ensemble_nn_weight": selected_weight,
            "config": {
                "window_size": config.WINDOW_SIZE,
                "window_stride": config.WINDOW_STRIDE,
                "cwt_scales": config.CWT_NUM_SCALES,
                "td_features": config.TD_FEATURES,
                "ensemble_weight_candidates": config.ENSEMBLE_WEIGHT_CANDIDATES,
                "batch_size": config.BATCH_SIZE,
                "num_epochs": config.NUM_EPOCHS,
                "torch_seeds": config.TORCH_SEEDS,
                "test_fraction": config.TEST_FRACTION,
                "internal_val_fraction": config.INTERNAL_VAL_FRACTION,
            },
            "dataset_counts_files": {k: len(v) for k, v in manifests.items()},
            "aggregate_accuracy": aggregate,
            "per_seed": per_seed_results,
        }, f, indent=2, default=_json_default)

    _plot_training_curves(per_seed_results)
    _plot_confusion_matrices(per_seed_results, aggregate)
    _write_run_log(manifests, aggregate, per_seed_results, n_params, time.time() - t0,
                    best_classical_name, classical_search_results,
                    selected_weight, weight_search_results)

    print(f"\nDone in {time.time() - t0:.1f}s. See {config.OUTPUT_DIR}/")


def _matched_subset_result(seed_nn, rf_proba, data, selected_weight):
    """Ensemble accuracy on just the baseline test files recorded on the same
    setup as this subject's attacks (config.ATTACK_MATCHED_SESSIONS). Returns
    None for subjects recorded entirely on one setup, where the overall test
    number already is the matched number."""
    if not config.ATTACK_MATCHED_SESSIONS:
        return None
    _, _, y, wm = data["test"]
    mask = wm["session"].isin(config.ATTACK_MATCHED_SESSIONS).to_numpy()
    if not mask.any():
        return None
    ens = selected_weight * seed_nn["nn_proba"]["test"] + (1 - selected_weight) * rf_proba["test"]
    return accuracy_from_proba(ens[mask], y[mask], wm[mask].reset_index(drop=True))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _aggregate_seeds(per_seed_results):
    aggregate = {c: {} for c in COMPONENTS}
    for comp in COMPONENTS:
        for split in EVAL_SPLITS:  # note: "test_matched" is reported separately, not aggregated here
            accs = [sr["results"][comp][split]["overall_accuracy"] for sr in per_seed_results]
            maj_accs = [sr["results"][comp][split]["file_majority_accuracy"] for sr in per_seed_results]
            aggregate[comp][split] = {
                "mean": float(np.mean(accs)), "std": float(np.std(accs)), "per_seed": accs,
                "majority_mean": float(np.mean(maj_accs)), "majority_std": float(np.std(maj_accs)),
                "majority_per_seed": maj_accs,
            }
    return aggregate


def _plot_training_curves(per_seed_results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for sr in per_seed_results:
        h = sr["history"]
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["val_loss"], alpha=0.8, label=f"seed {sr['seed']} val")
        axes[1].plot(epochs, h["val_acc"], alpha=0.8, label=f"seed {sr['seed']} val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(fontsize=8)
    axes[0].set_title("EMGNetFusion validation loss per seed")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend(fontsize=8)
    axes[1].set_title("EMGNetFusion validation accuracy per seed")
    fig.suptitle(f"EMGNetFusion training on {config.SUBJECT} baseline train_pool")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "training_curve.png", dpi=150)
    plt.close(fig)


def _plot_confusion_matrices(per_seed_results, aggregate):
    """Ensemble confusion matrices (the headline model), summed across seeds."""
    ncols = min(4, len(EVAL_SPLITS))
    nrows = -(-len(EVAL_SPLITS) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.8 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(EVAL_SPLITS):]:
        ax.axis("off")
    for ax, split in zip(axes, EVAL_SPLITS):
        cm_sum = np.sum([np.array(sr["results"]["ensemble"][split]["confusion_matrix"])
                          for sr in per_seed_results], axis=0)
        cm_norm = cm_sum / np.clip(cm_sum.sum(axis=1, keepdims=True), 1, None)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="viridis", ax=ax,
                    xticklabels=config.GESTURES, yticklabels=config.GESTURES,
                    cbar=False, vmin=0, vmax=1)
        a = aggregate["ensemble"][split]
        ax.set_title(f"{DISPLAY_NAME[split]}\nwindow acc={a['mean']:.3f}±{a['std']:.3f}\n"
                     f"file-majority acc={a['majority_mean']:.3f}±{a['majority_std']:.3f}", fontsize=9)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"{config.SUBJECT}: ensemble (headline model) confusion matrices", y=1.04)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def _write_run_log(manifests, aggregate, per_seed_results, n_params, elapsed_s,
                    best_classical_name, classical_search_results,
                    selected_weight, weight_search_results):
    lines = []
    lines.append(f"# EMGNetFusion + {best_classical_name} ensemble on {config.SUBJECT} (dataset_cleaned) -- run log\n")
    lines.append(f"EMGNetFusion parameters: {n_params}\n")
    lines.append(f"Seeds: {config.TORCH_SEEDS} (same data split every seed, only "
                 f"NN init/shuffling varies; the classical model is deterministic given the split)\n")
    lines.append(f"Ensemble weight: {selected_weight:.2f} on EMGNetFusion, "
                 f"{1 - selected_weight:.2f} on {best_classical_name} "
                 f"(selected by validation-only search, see below)\n")
    lines.append(f"Run time: {elapsed_s:.1f}s\n")

    lines.append("\n## Ensemble weight search (scored on the internal val split only, never test/attack)\n")
    lines.append(f"Winner: **{selected_weight:.2f}** NN weight\n")
    lines.append("| NN weight | Val window accuracy (mean across seeds) | Per-seed |")
    lines.append("|---|---|---|")
    for r in weight_search_results:
        marker = " **(selected)**" if r["weight"] == selected_weight else ""
        per_seed_str = ", ".join(f"{a:.3f}" for a in r["val_window_acc_per_seed"])
        lines.append(f"| {r['weight']:.2f}{marker} | {r['val_window_acc_mean']:.4f} | {per_seed_str} |")

    lines.append("\n## Classical-model search (grouped 3-fold CV by source file, train split only)\n")
    lines.append(f"Winner: **{best_classical_name}**\n")
    lines.append("| Model family | Best grouped-CV accuracy | Best hyperparameters |")
    lines.append("|---|---|---|")
    for name, res in classical_search_results.items():
        marker = " **(selected)**" if name == best_classical_name else ""
        lines.append(f"| {name}{marker} | {res['best_cv_accuracy']:.4f} | `{res['best_params']}` |")

    lines.append("\n## Dataset snapshot (file counts, for provenance)\n")
    lines.append("| Split | Files | Role |")
    lines.append("|---|---|---|")
    roles = {
        "train": "trained on (80% baseline train_pool minus internal val)",
        "val": "internal validation, used only for epoch/checkpoint selection",
        "test": "held out entirely from training -- the headline 20% baseline accuracy",
    }
    for name, mdf in manifests.items():
        role = roles.get(name, "held out entirely, evaluated once after training")
        lines.append(f"| {name} | {len(mdf)} | {role} |")

    lines.append(f"\n## Headline accuracy: Ensemble (EMGNetFusion + {best_classical_name}), mean +/- std\n")
    lines.append("Per-window accuracy scores every 100-sample window independently. "
                 "Per-file majority-vote accuracy applies the smoothing the paper itself "
                 "describes in Appendix A -- one file is one gesture repetition, so voting "
                 "across its windows cancels single-window noise. Both are reported.\n")
    lines.append("| Condition | Windows | Files | Window accuracy | File-majority accuracy |")
    lines.append("|---|---|---|---|---|")
    for split in EVAL_SPLITS:
        n = per_seed_results[0]["results"]["ensemble"][split]["n_windows"]
        nf = per_seed_results[0]["results"]["ensemble"][split]["n_files"]
        a = aggregate["ensemble"][split]
        per_seed_str = ", ".join(f"{x:.3f}" for x in a["per_seed"])
        maj_per_seed_str = ", ".join(f"{x:.3f}" for x in a["majority_per_seed"])
        lines.append(f"| {DISPLAY_NAME[split]} | {n} | {nf} | "
                     f"{a['mean']:.4f} +/- {a['std']:.4f} ({per_seed_str}) | "
                     f"{a['majority_mean']:.4f} +/- {a['majority_std']:.4f} ({maj_per_seed_str}) |")

    matched = [sr["results"]["ensemble"].get("test_matched") for sr in per_seed_results]
    if all(m is not None for m in matched):
        w = [m["overall_accuracy"] for m in matched]
        fm = [m["file_majority_accuracy"] for m in matched]
        lines.append("\n## Baseline accuracy restricted to the attack-matched setup\n")
        lines.append(f"Sessions: {', '.join(sorted(config.ATTACK_MATCHED_SESSIONS))} "
                     f"({matched[0]['n_files']} of the {per_seed_results[0]['results']['ensemble']['test']['n_files']} "
                     f"baseline test files). These were recorded on the same setup as this subject's "
                     f"attack sessions, so baseline-vs-attack is a controlled comparison only within "
                     f"this group. The overall baseline number above mixes both setups.\n")
        lines.append("| Metric | Accuracy (mean +/- std) | Per-seed |")
        lines.append("|---|---|---|")
        lines.append(f"| Window | {np.mean(w):.4f} +/- {np.std(w):.4f} | " + ", ".join(f"{x:.3f}" for x in w) + " |")
        lines.append(f"| File-majority | {np.mean(fm):.4f} +/- {np.std(fm):.4f} | " + ", ".join(f"{x:.3f}" for x in fm) + " |")

    lines.append("\n## Component comparison (file-majority accuracy)\n")
    lines.append(f"| Condition | EMGNetFusion (CNN+TD) | {best_classical_name} (TD only) | Ensemble |")
    lines.append("|---|---|---|---|")
    for split in EVAL_SPLITS:
        row = [f"{aggregate[c][split]['majority_mean']:.4f} +/- {aggregate[c][split]['majority_std']:.4f}"
               for c in COMPONENTS]
        lines.append(f"| {DISPLAY_NAME[split]} | " + " | ".join(row) + " |")

    lines.append("\n## Per-class accuracy, ensemble (mean across seeds)\n")
    lines.append("| Gesture | " + " | ".join(DISPLAY_NAME[s] for s in EVAL_SPLITS) + " |")
    lines.append("|---|" + "---|" * len(EVAL_SPLITS))
    for gesture in config.GESTURES:
        row = []
        for split in EVAL_SPLITS:
            vals = [sr["results"]["ensemble"][split]["per_class_accuracy"][gesture] for sr in per_seed_results]
            row.append(f"{np.nanmean(vals):.3f}")
        lines.append(f"| {gesture} | " + " | ".join(row) + " |")

    with open(config.OUTPUT_DIR / "run_log.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
