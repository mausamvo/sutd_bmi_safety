"""Same recordings, different sample rates: does 300 Hz beat 120 Hz?

Only fast-setup recordings are used (~345-390 rows/s), so every rate below is
derived from the SAME files by downsampling. Conditions:
  native     -- as recorded (~345-390 Hz)
  300        -- polyphase resample to 300 Hz (anti-aliased)
  120        -- polyphase resample to 120 Hz (anti-aliased), what the pipeline does
  120_naive  -- keep every k-th sample with no filter, which is what a slower
                ADC loop with no anti-alias filter does in hardware
Windows always cover the same time (0.833 s; 0.5 s for 1-s recordings) with
the same time stride, so only the sampling density changes. Model: ExtraTrees
with fixed settings (no tuning per condition). Evaluation: file-grouped,
gesture-stratified 5-fold CV, 3 shuffles. S2 also: train on all fast
baselines, score its fast attack sessions.
"""
import json, re, sys, warnings
from math import gcd
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import resample_poly
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # new_ml_code/
from src import config
from src.features import window_to_td_features

ROOT = config.DATA_ROOT
G = config.GESTURES
RE = re.compile(r"_(" + "|".join(G) + r")_set(\d+)_adc\.csv$")
CONDS = ["native", "300", "120", "120_naive"]

SETS = {
    "S2 (24 Apr, 3 s)": dict(subject="S2", win_s=100 / 120, baseline=["baseline_data_1007033_24_april", "baseline_data_2_1007033_24_april"],
                             attacks={"50 Hz": ["attack_50freq_1_1007033_24_april", "attack_50freq_2_1007033_24_april"],
                                      "laser": ["attack_laser_1_1007033_24_april", "attack_laser_2_1007033_24_april"]}),
    "S4 (yx4-5, 3 s)": dict(subject="S4", win_s=100 / 120, baseline=["baseline_yx4", "baseline_yx5"], attacks={}),
    "S4 (yx1-3, 1 s)": dict(subject="S4", win_s=0.5, baseline=["baseline_yx1", "baseline_yx2", "baseline_yx3"], attacks={}),
    "S5 (zen1-5, 1 s)": dict(subject="S5", win_s=0.5, baseline=[f"baseline_zen{i}" for i in range(1, 6)], attacks={}),
    "S3 (wj1-5, 1 s)": dict(subject="S3", win_s=0.5, baseline=[f"baseline_wj{i}" for i in range(1, 6)], attacks={}),
}
STRIDE_S = 5 / 120


def load(path):
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["Timestamp"])
    rate = len(df) / (ts.iloc[-1] - ts.iloc[0]).total_seconds()
    return df[config.CHANNEL_COLUMNS].to_numpy(np.float32).T, rate


def convert(sig, rate, cond):
    if cond == "native":
        return sig, rate
    target = 300.0 if cond == "300" else 120.0
    if cond == "120_naive":
        idx = np.round(np.arange(0, sig.shape[1], rate / target)).astype(int)
        idx = idx[idx < sig.shape[1]]
        return sig[:, idx], target
    up, down = int(target), int(round(rate)); g = gcd(up, down)
    return resample_poly(sig, up // g, down // g, axis=1).astype(np.float32), target


def features(files, cond, win_s):
    X, y, grp = [], [], []
    for path, gesture in files:
        sig, rate = load(path)
        s, r = convert(sig, rate, cond)
        config.EFFECTIVE_SAMPLE_RATE_HZ = r   # frequency features use the true rate
        w, st = int(round(win_s * r)), max(1, int(round(STRIDE_S * r)))
        for a in range(0, s.shape[1] - w + 1, st):
            X.append(window_to_td_features(s[:, a:a + w])); y.append(G.index(gesture)); grp.append(str(path))
    return np.array(X), np.array(y), np.array(grp)


def files_of(subject, sessions):
    out = []
    for sess in sessions:
        for f in sorted((ROOT / subject / sess).glob("*_adc.csv")):
            out.append((f, RE.search(f.name).group(1)))
    return out


def model():
    return ExtraTreesClassifier(n_estimators=300, max_features=0.5, max_depth=40, min_samples_leaf=1,
                                class_weight="balanced", random_state=42, n_jobs=-1)


def score(y, p, grp):
    win = float((y == p).mean())
    d = pd.DataFrame({"g": grp, "y": y, "p": p})
    fm = float(d.groupby("g").apply(lambda t: t.p.mode().iloc[0] == t.y.iloc[0]).mean())
    return win, fm


results = {}
for name, cfg in SETS.items():
    base = files_of(cfg["subject"], cfg["baseline"])
    rates = [load(f)[1] for f, _ in base[:10]]
    res = {"n_files": len(base), "native_rate": float(np.median(rates)), "win_s": cfg["win_s"], "cv": {}, "attacks": {}}
    for cond in CONDS:
        X, y, grp = features(base, cond, cfg["win_s"])
        accs = []
        for rep in range(3):
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
            P = np.empty_like(y)
            for tr, te in cv.split(X, y, grp):
                P[te] = model().fit(X[tr], y[tr]).predict(X[te])
            accs.append(score(y, P, grp))
        a = np.array(accs)
        res["cv"][cond] = dict(win=a[:, 0].mean(), win_sd=a[:, 0].std(), fm=a[:, 1].mean(), fm_sd=a[:, 1].std(),
                               windows=int(len(y)), win_len=int(round(cfg["win_s"] * (res["native_rate"] if cond == "native" else (300 if cond == "300" else 120)))))
        if cfg["attacks"]:
            clf = model().fit(X, y)
            for an, sess in cfg["attacks"].items():
                Xa, ya, ga = features(files_of(cfg["subject"], sess), cond, cfg["win_s"])
                res["attacks"].setdefault(an, {})[cond] = dict(zip(["win", "fm"], score(ya, clf.predict(Xa), ga)))
        print(f"{name:20s} {cond:10s} CV win {res['cv'][cond]['win']:.4f}±{res['cv'][cond]['win_sd']:.4f}  fm {res['cv'][cond]['fm']:.4f}"
              + "".join(f"  | {an}: {res['attacks'][an][cond]['win']:.4f}/{res['attacks'][an][cond]['fm']:.4f}" for an in cfg["attacks"]),
              flush=True)
    results[name] = res
json.dump(results, open(Path(__file__).with_suffix(".json"), "w"), indent=1)
