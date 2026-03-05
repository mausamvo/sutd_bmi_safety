import subprocess
import sys
import os
import shutil
import re
import argparse

DEFAULT_NUM_ITERATIONS = 50
DEFAULT_DATA_PATH = r"..\sutd_bmi_safety_data\combined.csv"
DEFAULT_UNSEEN_CSV = r"..\sutd_bmi_safety_data\unseen\*.csv"

MODEL_FILE = "semg_mlp.pth"
ENCODER_FILE = "label_encoder_mlp.pkl"
BEST_MODEL_FILE = "semg_mlp_best.pth"
BEST_ENCODER_FILE = "label_encoder_mlp_best.pkl"


def parse_val_acc(train_stdout: str):
    # Parses: "Best model saved with accuracy: 95.00%"
    m = re.search(r"Best model saved with accuracy:\s*([\d.]+)%", train_stdout)
    return float(m.group(1)) / 100.0 if m else None


def parse_unseen_acc(infer_stdout: str):
    # Prefer "Overall Metrics" block if present
    m = re.search(r"Overall Metrics:\s*[\r\n]+.*?Accuracy:\s*([\d.]+)", infer_stdout, re.DOTALL)
    if m:
        return float(m.group(1))
    # Fallback: last occurrence of "Accuracy:"
    all_matches = re.findall(r"Accuracy:\s*([\d.]+)", infer_stdout)
    return float(all_matches[-1]) if all_matches else None


def tail(s: str, n: int = 800):
    if not s:
        return ""
    return s[-n:] if len(s) > n else s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=DEFAULT_NUM_ITERATIONS)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--unseen_csv", default=DEFAULT_UNSEEN_CSV)
    parser.add_argument("--feature_preset", default="baseline")
    parser.add_argument("--timeout_sec", type=int, default=600)
    args = parser.parse_args()

    best_val_accuracy = -1.0
    best_iteration = -1
    all_results = []

    # ------------------------------
    # Iterative training (select by validation ONLY)
    # ------------------------------
    for i in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {i+1}/{args.iterations}")
        print(f"{'='*60}")

        # Remove current model artifacts so training script doesn't exit early
        for f in [MODEL_FILE, ENCODER_FILE]:
            if os.path.exists(f):
                os.remove(f)

        # Run training
        train_cmd = [
            sys.executable, "semg_train_mlp.py",
            "--data", args.data,
            "--feature_preset", args.feature_preset,
        ]

        try:
            train_result = subprocess.run(
                train_cmd,
                capture_output=True,
                text=True,
                timeout=args.timeout_sec
            )
        except subprocess.TimeoutExpired:
            print(f"Training timed out after {args.timeout_sec}s")
            continue

        print(tail(train_result.stdout, 1000))

        if train_result.returncode != 0:
            print("Training failed:")
            print(train_result.stderr)
            continue

        # Parse validation accuracy from training output
        val_acc = parse_val_acc(train_result.stdout)
        if val_acc is None:
            print("Could not parse validation accuracy from training output.")
            continue

        all_results.append({
            "iteration": i + 1,
            "val_acc": val_acc,
        })

        print(f">>> Iteration {i+1}: Val Acc = {val_acc:.4f} (best so far: {best_val_accuracy:.4f})")

        # Select best model by validation accuracy only
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_iteration = i + 1

            if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
                shutil.copy(MODEL_FILE, BEST_MODEL_FILE)
                shutil.copy(ENCODER_FILE, BEST_ENCODER_FILE)
                print(f">>> NEW BEST by validation! Saved as {BEST_MODEL_FILE}")
            else:
                print("Warning: trained model files not found to copy.")

    # ------------------------------
    # Summary of training runs
    # ------------------------------
    print(f"\n{'='*60}")
    print("SEARCH COMPLETE - TRAINING SUMMARY")
    print(f"{'='*60}")

    if not all_results:
        print("No successful training iterations.")
        return

    print(f"\nFeature preset: {args.feature_preset}")
    print(f"{'Iter':<6} {'Val Acc':<12} {'Best?'}")
    print("-" * 30)
    for r in all_results:
        marker = "<<<" if r["iteration"] == best_iteration else ""
        print(f"{r['iteration']:<6} {r['val_acc']:<12.4f} {marker}")

    print(f"\nBest iteration (by validation): {best_iteration}")
    print(f"Best validation accuracy:       {best_val_accuracy:.2%}")
    print(f"Saved model:                   {BEST_MODEL_FILE}")
    print(f"Saved encoder:                 {BEST_ENCODER_FILE}")

    # ------------------------------
    # Final unseen evaluation (run ONCE)
    # ------------------------------
    if not (os.path.exists(BEST_MODEL_FILE) and os.path.exists(BEST_ENCODER_FILE)):
        print("\nBest model artifacts not found. Skipping final unseen evaluation.")
        return

    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON UNSEEN DATA (RUN ONCE)")
    print(f"{'='*60}")

    # Copy best artifacts into the names expected by inference script
    shutil.copy(BEST_MODEL_FILE, MODEL_FILE)
    shutil.copy(BEST_ENCODER_FILE, ENCODER_FILE)

    infer_cmd = [
        sys.executable, "semg_infer_mlp.py",
        "--mode", "offline",
        "--csv", args.unseen_csv,
        "--feature_preset", args.feature_preset,
    ]

    try:
        infer_result = subprocess.run(
            infer_cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout_sec
        )
    except subprocess.TimeoutExpired:
        print(f"Final unseen inference timed out after {args.timeout_sec}s")
        return

    print(infer_result.stdout)
    if infer_result.returncode != 0:
        print("Final inference failed:")
        print(infer_result.stderr)
        return

    unseen_acc = parse_unseen_acc(infer_result.stdout)
    if unseen_acc is not None:
        print(f"\nFinal unseen accuracy: {unseen_acc:.4f}")
    else:
        print("\nCould not parse final unseen accuracy (but detailed output printed above).")


if __name__ == "__main__":
    main()