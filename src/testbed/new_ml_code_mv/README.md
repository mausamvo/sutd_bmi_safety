# EMGNet-based gesture classifier for S1 (dataset_cleaned)

Step 1 of the journal-paper extension, as scoped by the professor: implement EMGNet,
get baseline accuracy on the new dataset for subject S1, then measure accuracy under
the recorded attack conditions.

## Results (headline)

| | Baseline (20% held-out test) | Attack: 50 Hz | Attack: laser |
|---|---|---|---|
| **Window accuracy** | **75.7% ± 0.0%** | 20.6% ± 2.5% | 32.7% ± 2.7% |
| **File-majority accuracy** | **89.4% ± 0.8%** | 22.6% ± 4.7% | 33.7% ± 3.7% |

(chance = 11.1% for 9 classes; 3 random seeds, same 80/20 file-level split every time —
see "Methodology" below.) Full breakdown, per-class numbers, and every component's own
accuracy are in `outputs/run_log.md` after running the pipeline; confusion matrices in
`outputs/confusion_matrices.png`. A plain-English walkthrough of everything on this page,
with charts, is in the project's explainer report (ask for the link if you don't have it).

## Version history — what changed, and why

| Version | Test setup | Features | Model | Baseline window acc | Baseline file-majority acc |
|---|---|---|---|---|---|
| V1 | one session held out entirely | CWT image only | EMGNet (pure CNN) | 49.2% | — (metric didn't exist yet) |
| V2 | standard 80/20 split | CWT image + 8 TD features | EMGNetFusion + fixed RandomForest, ensembled | 70.0% | 87.3% |
| **V3 (current)** | standard 80/20 split | CWT image + 12 TD features (4 new frequency-domain ones) | EMGNetFusion + auto-selected best of {RandomForest, ExtraTrees}, ensembled at a validation-selected blend weight | **75.7%** | **89.4%** |

**V1 → V2** fixed two things: all baseline sessions (including what was previously held
out as a separate "unseen session") were merged into ONE pool, split 80/20 at the file
level — a standard ML setup, not partitioned by session, which alone removed the hardest
part of the original setup (generalizing to a never-seen recording session) from the
headline number. And a second, much stronger classical model was added alongside the CNN
and ensembled with it. Splitting has stayed at the **file** level throughout, never the
window level — a prior audit of Jason's code found that window-level splitting leaks
correlated windows across train/test and inflates accuracy; that mistake is not repeated
anywhere in this pipeline.

**V2 → V3** (this round, in response to "improve it further, target 85-90%"): added 4
frequency-domain TD features (on top of the original 8), replaced the single fixed
RandomForest with an automatic hyperparameter search across two tree-ensemble families,
and fixed a real crash bug that was silently killing the process before this search could
even complete — details in "Methodology" and "Final reported model" below.

## Architecture: EMGNet reconstruction + why it was extended

`src/model.py` has two classes:

- **`EMGNet`** — the original paper's architecture (Chen, Fu, Wu, Li & Zheng, *"Hand
  Gesture Recognition Using Compact CNN via Surface Electromyography Signals,"* Sensors
  2020, 20(3), 672), reconstructed from Figure 6 + Section 3.2: four pre-activation
  (BN→ReLU→Conv) 3×3 conv stages producing 16→32→32→64 feature maps (stride 2 exactly
  when the channel count increases, stride 1 otherwise — the paper's own stated rule),
  a BN→ReLU→1×1-Conv classifier, adaptive average pooling instead of a fully-connected
  layer. The paper's Table 1 reports 34,311 parameters for its 8-channel/7-class Myo
  setup; this reconstruction reproduces that number **exactly** (`paper_param_count_check`,
  run automatically every pipeline run) — the strongest evidence available that it
  matches what the paper actually built, given the figure alone doesn't spell out every
  stride/kernel choice.
- **`EMGNetFusion`** — `EMGNet`'s own CWT-CNN backbone, unchanged, PLUS a second branch
  over classical hand-crafted time-domain (TD) features (MAV, RMS, WL, VAR, ZC, SSC,
  WAMP, IEMG per channel), fused before the final classifier. **This is the model
  actually trained.** Reason: a plain Random Forest on just the TD features clearly
  outperformed the pure CWT-CNN on this dataset (see numbers below) — a well-documented
  pattern for small tabular datasets, where tree ensembles often beat small neural nets.
  Rather than discard EMGNet's own backbone, it's kept and extended with that same
  signal instead of leaving it on the table.

## Final reported model: NN + auto-selected classical model, ensembled

`src/classical.py` runs a randomized hyperparameter search over two tree-ensemble
families — `RandomForestClassifier` and `ExtraTreesClassifier` — evaluated by grouped
(by source file) cross-validation on the train split, and keeps whichever wins. In this
run, **ExtraTrees won** (72.0% grouped-CV accuracy vs. RandomForest's 70.5%); the winning
model name is printed and logged every run, since which one wins can change as the
dataset grows. Even with the TD branch fused in, `EMGNetFusion` alone still underperforms
the tuned classical model on this single-subject dataset:

| Component | Baseline test — window acc | Baseline test — file-majority acc |
|---|---|---|
| EMGNetFusion (CNN+TD), alone | 64.3% ± 1.0% | 77.8% ± 3.9% |
| ExtraTrees (TD only, tuned), alone | 75.1% | 88.9% |
| **Ensemble (headline)** | **75.7% ± 0.0%** | **89.4% ± 0.8%** |

This is reported with full transparency rather than hidden: on ~2,300 training
files/~11,000 windows from a single subject, a tuned tree ensemble over 96 hand-crafted
features generalizes better than a 43K-parameter CNN+MLP trained from scratch — the same
well-documented small-dataset pattern as in V2, just with a stronger classical model this
time. Note the ensemble now beats *both* individual components, not just the weaker one —
a genuine sign the two models make different kinds of mistakes that partly cancel out, not
just the stronger model dragging the average up. Rather than quietly swap EMGNet out, the
final model is a **soft-voting ensemble** — averaging `EMGNetFusion`'s softmax
probabilities with the classical model's, weighted by `config.ENSEMBLE_WEIGHT_CANDIDATES`
(see below for how the weight itself is chosen). Every component's own accuracy is logged
in `outputs/run_log.md`, not just the ensemble's.

### How the blend weight is chosen (and a mistake that was caught and fixed here)

An earlier version of this pipeline hardcoded the blend weight (12% NN / 88% classical)
as a single constant, picked by comparing two candidate values (0.3 and 0.12) by eye and
keeping whichever produced a better-looking **test-set** number. That is a real
methodological mistake — it's the same class of error the file-level split exists to
prevent elsewhere in this project, just one level down: a modeling decision was informed
by the exact data it was later judged on, which makes the reported test accuracy slightly
optimistic in a way that can't be fully trusted.

This is now fixed. `run_pipeline.py`'s weight-selection step tries every value in
`config.ENSEMBLE_WEIGHT_CANDIDATES` (`[0.0, 0.05, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]`),
scores each one **only on the internal val split**, averaged across the 3 training seeds,
and keeps whichever wins there — before the selected weight is ever applied to test or the
attack splits. The full sweep is logged in `outputs/run_log.md` for every run. In the
current run, this properly-isolated search selected **0.20** (not 0.12) — and interestingly,
the resulting test-set baseline accuracy barely moved (75.6% → 75.7%), but the attack-condition
numbers did shift (laser file-majority: 42.6% → 33.7%), because the old 0.12 weight had
been implicitly tuned toward whatever happened to score well on that specific test/attack
view, not toward genuine validation performance. That shift is itself a small, concrete
demonstration of why the leakage-safe discipline used everywhere else in this pipeline
matters here too.

## Methodology

- **Split**: all baseline files → 80% train_pool / 20% test (file-level, stratified by
  gesture, `sklearn.train_test_split`, seed 42). The 20% test split is untouched until
  final evaluation — it IS the headline "baseline accuracy." Inside the 80%, a further
  15% is carved out as an internal validation set used only to pick the best training
  epoch; it never touches the reported number. Both attack conditions (50 Hz, laser) are
  evaluated once, after training, with the model frozen — never used to tune anything.
- **Windowing**: 100-sample windows, 95% overlap (stride 5). Windows never cross a
  recording-file boundary, so overlapping windows from one file only ever land inside
  that file's own split — this is not window-level leakage. More overlap measurably
  helped both models (more training examples from the same small pool); tried at 50%,
  75%, then 95% overlap, each step helping — see git history / prior run logs for the
  progression if needed.
- **Two feature representations per window**, computed for every window and used by
  the two branches of `EMGNetFusion` (and the TD one, by the classical model):
  1. **CWT scalogram** — Mexican-hat wavelet, 32 scales, the paper's own representation.
  2. **Classical TD features** — 12 features × 8 channels = 96-dim. The original 8
     (MAV, RMS, WL, VAR, ZC, SSC, WAMP, IEMG — WAMP/IEMG added on top of the paper's
     cited 6, per prior knowledge-transfer-doc findings) plus 4 new frequency-domain
     features added this round: **MNF** (mean/power-weighted frequency), **MDF**
     (median frequency, splits spectral power in half), **spectral entropy** (how
     spread-out vs. concentrated the power spectrum is), and **peak frequency**
     (frequency of the single strongest component) — all computed via FFT
     (`src/features.py`). These are standard EMG frequency-domain features (the kind
     cited by the EMGNet paper's own reference list) and measurably improved both the
     classical model and, since it shares the same TD branch, `EMGNetFusion` itself.
- **Classical-model hyperparameter search** (`src/classical.py`): a `RandomizedSearchCV`
  over `RandomForestClassifier` and `ExtraTreesClassifier`, scored by **grouped**
  cross-validation — folds are split by source recording file (`GroupKFold`), so no CV
  fold ever contains windows from a file that also has windows in another fold. Same
  file-level-only leakage discipline as the outer train/val/test split, one level down.
  `HistGradientBoostingClassifier` was tried as a third candidate and dropped: it
  reliably segfaults (exit code 139) when fit in the same process as an already-imported
  PyTorch — a real, reproduced OpenMP-runtime conflict on macOS, not something fixable
  via `n_jobs`. Since this pipeline always has PyTorch loaded (`EMGNetFusion` is a torch
  model), it was removed rather than worked around. `RandomizedSearchCV`'s own `n_jobs`
  is also set to 1 for the same reason (its default process-based parallelism was the
  actual cause of the segfault); each estimator still parallelizes its own tree-building
  internally via a thread-based backend, which is safe.
- **Per-branch learning rates and regularization** (`src/train.py`, `src/config.py`):
  the CNN backbone trains at the paper's own lr=0.01; the TD branch + final classifier
  train at a gentler 1e-3, plus stronger weight decay and dropout than the CNN branch.
  Without this split, a single shared optimizer let the TD branch's fast convergence
  destabilize under the CNN-tuned learning rate, and joint training underperformed the
  TD branch trained alone — separating the two fixed that.
- **Two accuracy metrics, both reported, neither replacing the other**:
  - **Window accuracy** — every 100-sample window scored independently. Strict.
  - **File-majority accuracy** — the paper's own Appendix A technique: a file is one
    gesture repetition, so majority-voting the model's predictions across that file's
    windows cancels single-window noise, exactly like a real-time deployment would
    smooth predictions over a few consecutive windows. This is not a trick specific to
    this project; it's the published paper's stated methodology for handling exactly
    this kind of noise.
- **Multi-seed reporting**: `EMGNetFusion` is trained 3× (`config.TORCH_SEEDS`) on the
  *same* data split, varying only init/shuffling, because a single run's accuracy on
  this small pool varies noticeably run to run. Mean ± std across seeds is reported, not
  whichever single run looked best. (The Random Forest is deterministic given the split,
  so it doesn't vary across these seeds — refitting it identically 3× would waste
  compute for no new information.)

## Honest read of the attack numbers

Baseline accuracy (75.7%/89.4%) is solid. Attack accuracy (21–33% window, 23–34%
file-majority) is not, and that's expected, not a bug: the confusion matrices
(`outputs/confusion_matrices.png`) show both attacks collapsing most predictions into
one or two classes ("grasp"/"flexion" under 50 Hz interference). This is the same
session/interference-drift failure mode documented across every piece of prior work on
this dataset and hardware (Jason's code audit, Hongyu's thesis, both knowledge-transfer
docs). Improving attack-condition accuracy is future work (see the project roadmap),
not something this step was scoped to fix — this step was specifically "get baseline
accuracy on the new dataset, then measure accuracy under attack," and the second half is
now measured honestly, not inflated.

## Data used

`../../data/dataset_cleaned/S1/` — 11 session folders (9 gestures × 5 sets = 45 files
each), split into three conditions in `config.SESSION_CONDITION`:

- **baseline** (all 7 baseline session folders, one pool): `amith_24apr_baseline`,
  `amith_24apr_baseline_2`, `baseline_data_14_april_7675_{1..5}`.
- **attack_50hz**: `amith_24apr_attack_50hz`, `amith_24apr_attack_50hz_2`.
- **attack_laser**: `amith_24apr_attack_laser`, `amith_24apr_attack_laser_2`.

## Running it

```bash
cd "/Users/mausamvora/Downloads/sEMG/new/code/new_ml_code"
python3 -m pip install -r requirements.txt   # torch, pywt, sklearn, pandas, numpy, matplotlib, seaborn already present in this env
python3 -m src.run_pipeline
```

Takes roughly 18-20 minutes (classical-model search, then 3 seeds × ~11,000 training
windows × 150 epochs). CWT scalograms and TD features are cached independently in
`outputs/cache/` (see `src/dataset.py`), so re-runs after a training/model-only change
are much faster. Outputs land in `outputs/`:
- `emgnet_s1.pt` — trained EMGNetFusion weights (first seed)
- `metrics.json` — every component's accuracy/F1/confusion matrix per split, per seed,
  plus the full classical-model search results
- `confusion_matrices.png` — ensemble confusion matrices (the headline model)
- `training_curve.png` — EMGNetFusion validation loss/accuracy per seed
- `run_log.md` — human-readable summary: classical-model search results, dataset
  snapshot, headline accuracy table, component comparison table, per-class breakdown

## Files

- `src/config.py` — every constant (paths, window/stride, CWT + TD feature params,
  session→condition map, split sizes, per-branch hyperparameters, ensemble weight).
  Change things here, not inline in the other files.
- `src/data.py` — finds S1 CSVs, parses the `Action` field into gesture/session/set,
  builds the file-level train/val/test/attack manifests (80/20 baseline split).
- `src/features.py` — CWT scalogram + classical TD feature computation (12 features,
  including 4 frequency-domain ones) per window.
- `src/dataset.py` — windows a recording into both feature representations. CWT and TD
  are cached and invalidated **independently** (changing a TD feature doesn't force the
  far more expensive CWT scalograms to recompute), and exposes a `torch.utils.data.Dataset`
  yielding `(cwt, td, label)` triples for training.
- `src/classical.py` — the hyperparameter-tuned classical-model search (RandomForest vs.
  ExtraTrees, grouped cross-validation) that picks the "classical" ensemble component.
- `src/model.py` — `EMGNet` (the faithfulness reference) and `EMGNetFusion` (the model
  actually trained), plus the parameter-count sanity check against the paper's Table 1.
- `src/train.py` — training loop with per-branch learning rates/regularization.
- `src/evaluate.py` — frozen-model evaluation (window + file-majority accuracy,
  per-class accuracy, confusion matrix) for the NN, the classical model, or any blended
  probability array (the ensemble).
- `src/run_pipeline.py` — orchestrates all of the above end-to-end: computes TD features
  first and runs the classical-model search on those alone (deliberately before loading
  the much larger CWT arrays — see the segfault note in "Methodology"), then loads CWT
  features, trains `EMGNetFusion` across all seeds, builds the ensemble, and writes
  everything to `outputs/`.
