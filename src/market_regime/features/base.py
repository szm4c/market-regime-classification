from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseFeatureBuilder(ABC):
    """Abstract base for feature builders used in regime pipelines."""

    @abstractmethod
    def build_features(
        self,
        df: pd.DataFrame,
        is_train: bool,
    ) -> tuple[np.ndarray, np.ndarray | None, pd.Index]:
        """Return (x, y, index) for given DataFrame."""
        ...
