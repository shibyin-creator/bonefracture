"""GridSearchCV hyperparameter tuning optimized for accuracy, F1, MCC, and ROC-AUC."""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import config

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Systematic GridSearchCV matching SPIN 2024 Section VI.

    Multi-metric scoring: test accuracy (refit), F1-score, MCC, and ROC-AUC.
    """

    def __init__(
        self,
        fast: bool = config.FAST_TUNING,
        cv_folds: int = 3,
        random_state: int = config.RANDOM_STATE,
        n_jobs: int = -1,
    ) -> None:
        self.fast = fast
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.grids = config.FAST_PARAM_GRIDS if fast else config.PAPER_PARAM_GRIDS
        self.search_log: list[dict[str, Any]] = []

    def _cv(self) -> StratifiedKFold:
        return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

    def _scoring(self) -> dict[str, Any]:
        return {
            "accuracy": "accuracy",
            "f1": "f1",
            "mcc": make_scorer(matthews_corrcoef),
            "roc_auc": "roc_auc",
        }

    def tune(
        self,
        name: str,
        estimator: Any,
        X,
        y,
        param_grid: Optional[dict] = None,
    ) -> tuple[Any, dict[str, Any]]:
        grid = param_grid or self.grids.get(name)
        if not grid:
            logger.info("No search grid for %s; returning baseline estimator.", name)
            estimator.fit(X, y)
            record = {"model": name, "best_params": {}, "best_cv_accuracy": None}
            self.search_log.append(record)
            return estimator, record

        search = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            scoring=self._scoring(),
            refit=config.TUNING_REFIT_METRIC,
            cv=self._cv(),
            n_jobs=self.n_jobs,
            verbose=0,
        )
        logger.info("GridSearchCV for %s with %s combinations.", name, self._count_combos(grid))
        search.fit(X, y)
        cvres = search.cv_results_
        best_idx = search.best_index_
        record = {
            "model": name,
            "best_params": search.best_params_,
            "best_cv_accuracy": float(cvres["mean_test_accuracy"][best_idx]),
            "best_cv_f1": float(cvres["mean_test_f1"][best_idx]),
            "best_cv_mcc": float(cvres["mean_test_mcc"][best_idx]),
            "best_cv_roc_auc": float(cvres["mean_test_roc_auc"][best_idx]),
        }
        self.search_log.append(record)
        logger.info("%s best params: %s (CV acc=%.4f)", name, record["best_params"], record["best_cv_accuracy"])
        return search.best_estimator_, record

    def tune_suite(self, models: dict[str, Any], X, y, skip: Optional[set[str]] = None) -> dict[str, Any]:
        skip = skip or {"Stacking Classifier"}
        tuned = {}
        for name, estimator in models.items():
            if name in skip:
                tuned[name] = estimator
                continue
            best_est, _ = self.tune(name, estimator, X, y)
            tuned[name] = best_est
        return tuned

    def log_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.search_log)

    @staticmethod
    def _count_combos(grid: dict) -> int:
        n = 1
        for values in grid.values():
            n *= max(1, len(list(values)))
        return n
