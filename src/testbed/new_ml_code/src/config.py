"""
All tunable constants for the EMGNet pipeline live here. Nothing in the other
modules should hardcode a path, window size, or hyperparameter -- change it here.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent                      # .../new/code/new_ml_code
SEMG_ROOT = PROJECT_ROOT.parent.parent               # .../sEMG
DATA_ROOT = SEMG_ROOT / "data" / "dataset_cleaned"

# Which subject this run analyses. Per the professor's direction, subjects are
# analysed ONE AT A TIME -- baseline vs attack within a single subject -- so
# there is no cross-subject pooling anywhere in this pipeline.
SUBJECT = "S3"

SUBJECT_DIR = DATA_ROOT / SUBJECT
OUTPUT_DIR = PROJECT_ROOT / "outputs" / SUBJECT
S1_DIR = DATA_ROOT / "S1"  # kept for backwards compatibility with older calls

# ---------------------------------------------------------------------------
# Signal columns -> EMGNet's 8 "channels".
# Our hardware logs an (activation, envelope) pair per physical electrode;
# we treat the 4 physical sensors x 2 signals = 8 input channels, matching
# the Myo armband's 8-channel input in the paper (see README for rationale).
# ---------------------------------------------------------------------------
CHANNEL_COLUMNS = [
    "Ch0 Act", "Ch0 Env",
    "Ch1 Act", "Ch1 Env",
    "Ch2 Act", "Ch2 Env",
    "Ch3 Act", "Ch3 Env",
]
NUM_CHANNELS = len(CHANNEL_COLUMNS)

GESTURES = [
    "nothing", "grasp", "flexion", "extension",
    "pronation", "supination", "open", "left", "right",
]
NUM_CLASSES = len(GESTURES)

# ---------------------------------------------------------------------------
# Session folders -> condition label, per subject.
# All of a subject's baseline sessions are ONE pool, split 80/20 at the file
# level for train/test like a standard ML setup. Attack sessions are never
# trained on -- they are scored once, at the end, with the model frozen.
# ---------------------------------------------------------------------------
SESSION_CONDITION_BY_SUBJECT = {
    "S1": {
        "amith_24apr_baseline": "baseline",
        "amith_24apr_baseline_2": "baseline",
        "baseline_data_14_april_7675": "baseline",
        "baseline_data_14_april_7675_2": "baseline",
        "baseline_data_14_april_7675_3": "baseline",
        "baseline_data_14_april_7675_4": "baseline",
        "baseline_data_14_april_7675_5": "baseline",
        "amith_24apr_attack_50hz": "attack_50hz",
        "amith_24apr_attack_50hz_2": "attack_50hz",
        "amith_24apr_attack_laser": "attack_laser",
        "amith_24apr_attack_laser_2": "attack_laser",
    },
    "S2": {
        # 14 April baselines were recorded on the SLOW setup (~121 Hz);
        # everything from 24 April -- both baselines and every attack -- was
        # recorded on the FAST setup (~344 Hz). Resampling (below) puts them
        # on a common rate; run_pipeline additionally reports the 24 April
        # baseline accuracy on its own, since that is the only baseline
        # recorded on the same setup as the attacks.
        "baseline_data_14_april_7033_1": "baseline",
        "baseline_data_14_april_7033_2": "baseline",
        "baseline_data_14_april_7033_3": "baseline",
        "baseline_data_14_april_7033_4": "baseline",
        "baseline_data_14_april_7033_5": "baseline",
        "baseline_data_1007033_24_april": "baseline",
        "baseline_data_2_1007033_24_april": "baseline",
        "attack_50freq_1_1007033_24_april": "attack_50hz",
        "attack_50freq_2_1007033_24_april": "attack_50hz",
        "attack_laser_1_1007033_24_april": "attack_laser",
        "attack_laser_2_1007033_24_april": "attack_laser",
    },
    "S3": {
        # 14 April baselines were recorded on the FAST setup (~389 Hz) and are
        # only ~1 s long (~4 windows each after resampling); everything from
        # 24 April -- baselines and all four attack types -- was recorded on
        # the SLOW setup (~120 Hz) and is ~3 s long. Same handling as S2: one
        # baseline pool, plus a separately reported attack-matched baseline.
        "baseline_wj1": "baseline",
        "baseline_wj2": "baseline",
        "baseline_wj3": "baseline",
        "baseline_wj4": "baseline",
        "baseline_wj5": "baseline",
        "WJ_24apr": "baseline",
        "WJ_24apr_2": "baseline",
        "WJ_24apr_attack_50hz": "attack_50hz",
        "WJ_24apr_attack_50hz_2": "attack_50hz",
        "WJ_24apr_attack_laser": "attack_laser",
        "WJ_24apr_attack_laser_2": "attack_laser",
        # Incomplete session: 29 of 45 files (no open/left/right, supination
        # set 5 missing). Scored as-is; its accuracy covers 6 gestures only.
        "WJ_24apr_attack_5g": "attack_5g",
        "WJ_24apr_attack_5g_5cm": "attack_5g_5cm",
        "WJ_24apr_attack_5g_5cm_2": "attack_5g_5cm",  # run B split off below
    },
}
SESSION_CONDITION = SESSION_CONDITION_BY_SUBJECT[SUBJECT]

# Folders holding more than one recording run, split by filename epoch (ms):
# files at or after the cutoff get the override condition instead.
#
# S3 WJ_24apr_attack_5g_5cm_2 holds two complete 45-file runs ~6 min apart.
# Run A looks like every other session. Run B does not: signal activity is
# almost identical for all nine gestures (Act std 9.1-10.6 for every gesture,
# spread 5% vs ~25% in every other session) and it contains all 3 of S3's
# flat-channel files -- i.e. it looks like gestures were not being performed
# or the electrodes lost contact. It is scored as its own split so it can
# neither inflate nor deflate the 5G-5cm result; students have been asked.
FILE_CONDITION_OVERRIDES = {
    "S3": {"WJ_24apr_attack_5g_5cm_2": (1777001570000, "attack_5g_5cm_runB")},
}.get(SUBJECT, {})

# Fixed reporting order for attack conditions (only those present are used).
ATTACK_CONDITION_ORDER = ["attack_50hz", "attack_laser", "attack_5g",
                          "attack_5g_5cm", "attack_5g_5cm_runB"]
ATTACK_CONDITIONS = [c for c in ATTACK_CONDITION_ORDER
                     if c in set(SESSION_CONDITION.values())
                     or any(o[1] == c for o in FILE_CONDITION_OVERRIDES.values())]

# Sessions recorded on the same setup as this subject's attack sessions.
# Baseline-vs-attack is only a controlled comparison within this group, so
# its accuracy is reported separately as well as inside the overall number.
ATTACK_MATCHED_SESSIONS = {
    "S1": None,  # S1 is entirely one setup, so every baseline session matches
    "S2": {"baseline_data_1007033_24_april", "baseline_data_2_1007033_24_april"},
    "S3": {"WJ_24apr", "WJ_24apr_2"},
}[SUBJECT]

# Optional variant: train (and validate) ONLY on the attack-matched baseline
# sessions. The outer 80/20 split is still made on the full baseline pool
# first, so the test files -- and the attack files -- are exactly the same as
# in the default run; only the training/validation files change. Enable with
# the environment variable SEMG_TRAIN_ON=matched. Outputs go to their own
# folder (e.g. outputs/S3_matched_only/) so the default run is not overwritten.
import os as _os
TRAIN_ON_MATCHED_ONLY = _os.environ.get("SEMG_TRAIN_ON") == "matched"
if TRAIN_ON_MATCHED_ONLY and not ATTACK_MATCHED_SESSIONS:
    raise ValueError(f"{SUBJECT} has no ATTACK_MATCHED_SESSIONS to train on")
if TRAIN_ON_MATCHED_ONLY:
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / f"{SUBJECT}_matched_only"

# Non-recording folders/files present under a subject dir that must be skipped.
NON_SESSION_DIRS = {"visualization"}

# ---------------------------------------------------------------------------
# Windowing + CWT.
# Empirically, the cleaned S1 CSVs run at ~120 Hz effective sample rate
# (measured directly from timestamps -- see cleaning_report.md), not the
# nominal 500 Hz ADC rate. Window size/overlap tuned on validation accuracy.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Sample-rate normalisation.
#
# The recordings were made on two setups running at different speeds: a slow
# one (~120 rows/sec) and a fast one (~345-390 rows/sec). Some subjects have
# both -- S2's 14 April baselines are slow while ALL of its 24 April data,
# baselines and attacks alike, is fast. A fixed window of N rows would
# therefore cover completely different amounts of real time depending on
# which setup a file came from, and the frequency-domain features would be
# computed against the wrong rate.
#
# Every recording is therefore resampled to one common rate before windowing
# (see dataset.py::_resample_to_target). The target is the SLOW rate, because
# downsampling only discards information, while upsampling the slow files to
# match the fast ones would invent samples that were never recorded.
# scipy's polyphase resampler applies an anti-aliasing filter on the way down.
#
# After this step, EFFECTIVE_SAMPLE_RATE_HZ is true by construction for every
# file, rather than being an assumption that only held for S1.
TARGET_SAMPLE_RATE_HZ = 120.0
EFFECTIVE_SAMPLE_RATE_HZ = TARGET_SAMPLE_RATE_HZ

# Files already within this fraction of the target are left untouched, so the
# slow-setup recordings are not needlessly run through a resampler.
RESAMPLE_TOLERANCE = 0.05

WINDOW_SIZE = 100
WINDOW_STRIDE = 5            # 95% overlap -- more windows from the same pool.
                             # Still safe from leakage: splitting is file-level
                             # (data.py), and windows never cross a file
                             # boundary, so overlapping windows from one file
                             # only ever land within that file's own split.
CWT_WAVELET = "mexh"         # Mexican hat, same as the paper
CWT_NUM_SCALES = 32          # same as the paper
CWT_SCALES = list(range(1, CWT_NUM_SCALES + 1))
SPECTROGRAM_DOWNSAMPLE = 1   # full CWT resolution (see README tuning log)

# ---------------------------------------------------------------------------
# Classical hand-crafted time-domain features, computed per channel per
# window and fed into the model's second branch (see model.py::EMGNetFusion).
# WAMP (Willison Amplitude) and IEMG (Integrated EMG) added on top of the
# original paper-cited 6, per prior KT-doc findings that a richer TD feature
# set helps this exact hardware/dataset (Lee Wei Jie's `rf_enhanced` presets).
# ---------------------------------------------------------------------------
# MNF/MDF/spectral entropy/peak frequency are frequency-domain features
# (computed via FFT in features.py) added on top of the original 8 time-
# domain ones -- EMG literature (e.g. Phinyomark et al., cited by the
# EMGNet paper itself) documents these as discriminative for muscle
# contraction type, and they gave RandomForest a real accuracy gain on this
# dataset (see README's tuning log for the before/after numbers).
TD_FEATURES = ["MAV", "RMS", "WL", "VAR", "ZC", "SSC", "WAMP", "IEMG",
               "MNF", "MDF", "SPEC_ENTROPY", "PEAK_FREQ"]
WAMP_THRESHOLD = 1.0  # counts a sample-to-sample change as an activity event
                       # above this magnitude; ADC units are ~O(1-10) here.
NUM_TD_FEATURES = len(TD_FEATURES) * NUM_CHANNELS

# ---------------------------------------------------------------------------
# Split: baseline is 80% train_pool / 20% test (file-level, stratified by
# gesture) -- the 20% test split IS the headline "baseline accuracy". Inside
# the 80% train_pool, a further slice is held out as a validation set purely
# for epoch/checkpoint selection during training; it never contributes to
# the reported baseline number.
# ---------------------------------------------------------------------------
TEST_FRACTION = 0.2
INTERNAL_VAL_FRACTION = 0.15   # carved out of the 80% train_pool, not of the whole dataset
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Training hyperparameters (paper: Adam, lr 0.01 -> /10 at epoch 20 -> /10 at
# epoch 40, batch 128, 50 epochs, L2 = 1e-2 on the Myo dataset -- Table 2).
# Our training pool is far smaller than the paper's, so batch size is
# reduced and the schedule stretched accordingly (tuned on validation only).
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 150
INITIAL_LR = 0.01            # CNN backbone branch only, matches paper's Myo setting
TD_BRANCH_LR = 1e-3          # TD branch + classifier -- see README/train.py for why
                             # this needs to be lower than the CNN branch's lr
LR_DECAY_EPOCHS = (60, 110)
LR_DECAY_FACTOR = 10.0
CNN_WEIGHT_DECAY = 1e-2      # matches the paper's Myo L2 setting
CNN_DROPOUT = 0.0            # see README -- tried at 0.4, caused underfitting on this pool size
# The TD branch trains fast (its features are strong) and was observed to
# overfit hard once given its own learning rate (train accuracy -> ~90%
# while test stayed ~52%) -- stronger weight decay + dropout than the CNN
# branch brought that gap down considerably (see README tuning log).
TD_WEIGHT_DECAY = 1.5e-2
TD_DROPOUT = 0.15

# Runs training this many times with different torch/model-init seeds on
# the SAME data split, and reports mean +/- std -- accuracy on this small
# pool varies noticeably run-to-run with a single seed.
TORCH_SEEDS = [0, 1, 2]

# Per-window per-channel z-score normalization: tried, measured worse than
# leaving raw ADC magnitudes alone (see README). Off by default.
NORMALIZE_WINDOWS = False

# ---------------------------------------------------------------------------
# Final reported model: a soft-voting ensemble of EMGNetFusion (the deep
# CWT-CNN + TD network above) and the best of three hyperparameter-tuned
# classical (non-NN) models -- RandomForest, ExtraTrees, HistGradientBoosting
# -- trained on the same TD features (see src/classical.py). Why: on this
# dataset size, tree-ensemble models over the classical features alone
# clearly outperform the neural network (see README's tuning log for the
# side-by-side numbers) -- a well-documented pattern for small tabular
# datasets. Rather than discard the CNN, both models' predicted class
# probabilities are averaged (weighted by ENSEMBLE_NN_WEIGHT) and the
# ensemble's accuracy is the headline number; each component's own accuracy
# is still reported for transparency.
# ---------------------------------------------------------------------------
RF_RANDOM_STATE = 42  # random_state for every model in the classical-model search

# The NN's weight in the averaged probabilities (the classical model gets
# 1 - weight) is chosen by trying every value below and keeping whichever
# scores best -- but ONLY on the internal validation split (val), never on
# test or the attack splits, so the choice can't be informed by the numbers
# it's later judged on. See run_pipeline.py's weight-selection step.
#
# (History: an earlier version of this constant was a single hardcoded
# value, 0.12, picked by comparing two candidates -- 0.3 and 0.12 -- by
# eye, using their effect on the TEST split's accuracy. That was a real
# methodological mistake: a modeling decision was informed by the same
# split it was later scored on, which is the same class of error the
# leakage-safe file-level split exists to prevent elsewhere in this
# pipeline. This search replaces that with a proper validation-only choice.)
ENSEMBLE_WEIGHT_CANDIDATES = [0.0, 0.05, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
