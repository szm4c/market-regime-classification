from typing import Any, Optional, Self, Union

import numpy as np
from xgboost import XGBClassifier

from market_regime.models.base import BaseModel


class XGBTreeModel(BaseModel):
    """
    XGBoost multiclass classifier for regime labels {-1, 0, 1}.

    Internally maps:
        -1 -> 0
         0 -> 1
         1 -> 2
    """

    def __init__(
        self,
        # hyperparameters
        max_depth: int = 3,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        colsample_bynode: Optional[float] = None,
        learning_rate: float = 0.05,
        n_estimators: int = 500,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        max_delta_step: int = 0,
        # misc
        random_state: int = 333,
        n_jobs: int = -1,
        tree_method: str = "hist",
        verbosity: int = 0,
        eval_metric: str = "mlogloss",
        # passthrough params
        kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.max_depth = int(max_depth)
        self.min_child_weight = float(min_child_weight)
        self.gamma = float(gamma)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.colsample_bynode = (
            None if colsample_bynode is None else float(colsample_bynode)
        )
        self.learning_rate = float(learning_rate)
        self.n_estimators = int(n_estimators)
        self.reg_alpha = float(reg_alpha)
        self.reg_lambda = float(reg_lambda)
        self.max_delta_step = int(max_delta_step)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.tree_method = str(tree_method)
        self.verbosity = int(verbosity)
        self.eval_metric = str(eval_metric)
        self.kwargs: dict[str, Any] = dict(kwargs) if kwargs else {}
        # fitted state
        self.model_: Optional[XGBClassifier] = None
        self.fitted: bool = False
        # label mapping
        self._y_to_int: dict[int, int] = {-1: 0, 0: 1, 1: 2}
        self._int_to_y: dict[int, int] = {0: -1, 1: 0, 2: 1}

    @staticmethod
    def _ensure_2d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {x.shape}.")
        return x

    def _map_y(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).ravel()
        uniq = np.unique(y)
        allowed = np.array([-1, 0, 1])
        # allow subsets (some folds / windows might miss a class), but no other labels
        if not np.isin(uniq, allowed).all():
            raise ValueError(f"Expected labels in {{-1,0,1}}. Got: {uniq.tolist()}")
        return np.vectorize(self._y_to_int.get, otypes=[int])(y).astype(int)

    def _make_model(self) -> XGBClassifier:
        params: dict[str, Any] = dict(
            objective="multi:softprob",
            num_class=3,
            eval_metric=self.eval_metric,
            tree_method=self.tree_method,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            max_delta_step=self.max_delta_step,
        )

        if self.colsample_bynode is not None:
            params["colsample_bynode"] = self.colsample_bynode

        params.update(self.kwargs)
        return XGBClassifier(**params)

    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        x = self._ensure_2d(x)
        y_int = self._map_y(y)

        if x.shape[0] != y_int.shape[0]:
            raise ValueError("X and y have different number of rows.")

        self.model_ = self._make_model()
        self.model_.fit(x, y_int)

        self.fitted = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted or self.model_ is None:
            raise RuntimeError("Model must be fitted before predicting.")

        x = self._ensure_2d(x)
        pred_int = self.model_.predict(x).astype(int)
        return np.vectorize(self._int_to_y.get, otypes=[int])(pred_int).astype(int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted or self.model_ is None:
            raise RuntimeError("Model must be fitted before predicting.")

        x = self._ensure_2d(x)
        proba = np.asarray(self.model_.predict_proba(x))

        if proba.ndim != 2 or proba.shape[1] != 3:
            raise RuntimeError(f"Expected proba shape (n,3). Got {proba.shape}.")

        return proba
