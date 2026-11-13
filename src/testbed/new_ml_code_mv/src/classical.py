"""
Hyperparameter-tuned classical (non-neural-net) model on the TD features.

Tries two tree-ensemble families (RandomForest, ExtraTrees) with a
randomized hyperparameter search, evaluated by GROUPED cross-validation on
the train split -- grouped by source recording FILE (never by window), so no
CV fold ever contains windows from a file that also has windows in another
fold. This is the same file-level-only-ever leakage discipline used for the
outer train/val/test split (see data.py), applied one level down to the
search itself.

A third family, HistGradientBoostingClassifier, was tried and dropped: it
reliably segfaults (exit 139) when fit in the same process as an already-
imported PyTorch, even for one plain .fit() call with no search/CV involved
-- a known class of OpenMP-runtime conflict between PyTorch and
scikit-learn's Cython/OpenMP-parallel histogram-building code on macOS, not
something fixable via n_jobs or threading backend choice (unlike the
RandomizedSearchCV-level multiprocessing issue below, which IS fixable that
way). Since this pipeline always has PyTorch imported (EMGNetFusion is a
torch model), HistGradientBoosting was removed rather than worked around.

The best-scoring model overall (by mean CV accuracy) is refit on the full
train split and returned -- this becomes the "classical" component of the
final NN+classical ensemble in run_pipeline.py.
"""
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from . import config

_SEARCH_SPACE = {
    # RandomForest/ExtraTrees parallelize their own tree-building internally
    # with a THREADING joblib backend (no subprocesses), which is safe to
    # combine with PyTorch in the same process -- unlike RandomizedSearchCV's
    # own n_jobs (process-based), which is NOT (see fit_best_classical_model).
    "RandomForest": (
        RandomForestClassifier(random_state=config.RF_RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators": [300, 500, 800],
            "max_depth": [None, 15, 25, 40],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.3, 0.5],
            "class_weight": [None, "balanced"],
        },
    ),
    "ExtraTrees": (
        ExtraTreesClassifier(random_state=config.RF_RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators": [300, 500, 800],
            "max_depth": [None, 15, 25, 40],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.3, 0.5],
            "class_weight": [None, "balanced"],
        },
    ),
}


def fit_best_classical_model(
    X_td_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray,
    n_iter: int = 10, cv_splits: int = 3,
) -> Tuple[str, object, Dict]:
    """
    groups_train: one group id per window (its source file path) -- so
    GroupKFold never splits one file's windows across CV folds.
    Returns (best_model_name, fitted_best_estimator, all_search_results).
    """
    cv_folds = list(GroupKFold(n_splits=cv_splits).split(X_td_train, y_train, groups_train))

    search_results = {}
    best_name, best_estimator, best_score = None, None, -1.0
    for name, (estimator, param_dist) in _SEARCH_SPACE.items():
        search = RandomizedSearchCV(
            estimator, param_dist, n_iter=n_iter, cv=cv_folds,
            scoring="accuracy", random_state=config.RF_RANDOM_STATE,
            # n_jobs=1 (no joblib multiprocessing) is deliberate, not a typo:
            # this runs in the same process as PyTorch (imported elsewhere in
            # the pipeline), and joblib's multiprocess worker pool + PyTorch's
            # own OpenMP/BLAS threading segfaulted this process (exit code
            # 139) when combined -- a known class of conflict on macOS. Each
            # individual estimator is still fit with its own internal
            # n_jobs=-1 (see _SEARCH_SPACE) for tree-building parallelism
            # within a single process, so this isn't fully sequential.
            n_jobs=1, refit=True,
        )
        search.fit(X_td_train, y_train)
        search_results[name] = {
            "best_cv_accuracy": float(search.best_score_),
            "best_params": {k: (v if not isinstance(v, (np.integer, np.floating)) else v.item())
                            for k, v in search.best_params_.items()},
        }
        print(f"  [{name:22s}] best grouped-CV accuracy = {search.best_score_:.4f}  "
              f"params={search.best_params_}")
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_name = name
            best_estimator = search.best_estimator_

    print(f"  -> selected {best_name} (grouped-CV accuracy {best_score:.4f})")
    return best_name, best_estimator, search_results
