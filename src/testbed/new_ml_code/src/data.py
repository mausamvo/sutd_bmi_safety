"""
Enumerates the active subject's cleaned CSVs, parses each filename into (session, condition,
gesture, set number), and builds file-level manifests:

  train, val   -- carved out of the 80% baseline train_pool, val used only
                  for epoch/checkpoint selection during training
  test         -- the untouched 20% of the baseline pool. THIS is the
                  headline "baseline accuracy" split.
  one split per attack condition (config.ATTACK_CONDITIONS, e.g.
  attack_50hz, attack_laser) -- evaluated once, after training, frozen model.

Splitting is always done by whole recording file, never by window -- a
prior audit of Jason_code's ML pipeline found window-level splitting leaks
information across train/test because adjacent windows within one 3-5s
repetition are highly correlated. That constraint is kept; what changed
from the previous version is that ALL baseline sessions are now one pool
(no separate "unseen session" partition), split 80/20 like a standard ML
setup, per direction.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

_FILENAME_RE = re.compile(
    r"^(\d+)_.*_(" + "|".join(config.GESTURES) + r")_set(\d+)_adc\.csv$"
)


@dataclass(frozen=True)
class Recording:
    path: Path
    session: str
    condition: str
    gesture: str
    set_num: int


def _parse_filename(path: Path, session: str) -> Recording:
    m = _FILENAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not parse gesture/set from filename: {path.name}")
    epoch_ms, gesture, set_num = int(m.group(1)), m.group(2), int(m.group(3))
    condition = config.SESSION_CONDITION[session]
    override = config.FILE_CONDITION_OVERRIDES.get(session)
    if override is not None and epoch_ms >= override[0]:
        condition = override[1]
    return Recording(path=path, session=session, condition=condition,
                      gesture=gesture, set_num=set_num)


def enumerate_recordings(s1_dir: Path = config.SUBJECT_DIR) -> List[Recording]:
    recordings: List[Recording] = []
    for session_dir in sorted(s1_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        session = session_dir.name
        if session in config.NON_SESSION_DIRS:
            continue
        if session not in config.SESSION_CONDITION:
            raise ValueError(
                f"Unknown session folder '{session}' -- add it to "
                f"config.SESSION_CONDITION before running the pipeline."
            )
        for csv_path in sorted(session_dir.glob("*_adc.csv")):
            recordings.append(_parse_filename(csv_path, session))
    return recordings


def recordings_to_frame(recordings: List[Recording]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in recordings])


def build_manifests(s1_dir: Path = config.SUBJECT_DIR) -> Dict[str, pd.DataFrame]:
    df = recordings_to_frame(enumerate_recordings(s1_dir))

    pool = df[df.condition == "baseline"].reset_index(drop=True)

    # Outer split: 80% train_pool / 20% test. `test` is never touched again
    # until final evaluation -- this is the headline baseline accuracy.
    train_pool, test_df = train_test_split(
        pool,
        test_size=config.TEST_FRACTION,
        random_state=config.RANDOM_SEED,
        stratify=pool["gesture"],
    )

    # Optional variant (config.TRAIN_ON_MATCHED_ONLY): drop the non-matched
    # sessions from the train_pool only. `test` is left as it is, so both
    # variants are scored on the same files.
    if config.TRAIN_ON_MATCHED_ONLY:
        train_pool = train_pool[train_pool.session.isin(config.ATTACK_MATCHED_SESSIONS)]

    # Inner split: carve a validation slice out of the 80% train_pool, used
    # only to pick the best training epoch/checkpoint. Never touches `test`.
    train_df, val_df = train_test_split(
        train_pool,
        test_size=config.INTERNAL_VAL_FRACTION,
        random_state=config.RANDOM_SEED,
        stratify=train_pool["gesture"],
    )

    manifests = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
    for cond in config.ATTACK_CONDITIONS:
        manifests[cond] = df[df.condition == cond].reset_index(drop=True)

    unused = set(df.condition) - {"baseline"} - set(config.ATTACK_CONDITIONS)
    if unused:
        raise ValueError(f"Conditions {sorted(unused)} are not in config.ATTACK_CONDITION_ORDER")

    # Sanity: no file appears in more than one split.
    seen = {}
    for split_name, split_df in manifests.items():
        for p in split_df["path"]:
            if p in seen:
                raise AssertionError(
                    f"Leakage: {p} is in both '{seen[p]}' and '{split_name}'"
                )
            seen[p] = split_name

    return manifests


def load_recording(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    manifests = build_manifests()
    for name, mdf in manifests.items():
        print(f"{name:14s} {len(mdf):4d} files  " +
              ", ".join(f"{g}={c}" for g, c in mdf.gesture.value_counts().sort_index().items()))
