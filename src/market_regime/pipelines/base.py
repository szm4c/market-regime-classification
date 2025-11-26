from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Self
import numpy as np
import pandas as pd


class BaseFeatureBuilder(ABC):
    """Abstract base for feature builders used in regime pipelines."""

    @abstractmethod
    def build_features(
        self,
        df: pd.DataFrame,
        is_train: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], pd.Index]:
        """Return (x, y, index) for given DataFrame."""
        ...


class BaseEstimator(ABC):
    """Abstract base for estimators used in regime pipelines."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        """Fit estimator on features X and labels y."""
        ...

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels for features X."""
        ...

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities for features X."""
        ...


@dataclass  # init and repr
class RegimePipeline:
    """Pipeline combining a feature builder and an estimator."""

    feature_builder: BaseFeatureBuilder
    estimator: BaseEstimator

    def fit(self, df_train: pd.DataFrame) -> Self:
        """Fit pipeline on training DataFrame."""
        x_train, y_train, _ = self.feature_builder.build_features(
            df_train,
            is_train=True,
        )

        if y_train is None:
            raise ValueError("Training requires labels (y), but got None.")

        self.estimator.fit(x_train, y_train)

        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict labels for DataFrame df."""

        x, _, idx = self.feature_builder.build_features(df, is_train=False)

        y_pred = self.estimator.predict(x)

        return pd.Series(y_pred, index=idx, name="prediction")

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities for DataFrame df."""

        x, _, idx = self.feature_builder.build_features(df, is_train=False)
        proba = self.estimator.predict_proba(x)

        n_classes = proba.shape[1]
        if n_classes == 3:
            col_names = ["proba_-1", "proba_0", "proba_1"]
        else:
            raise Exception(f"Number of classes is wrong! Expected 3 got {n_classes}")

        return pd.DataFrame(proba, index=idx, columns=col_names)
