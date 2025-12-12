from abc import ABC, abstractmethod
from typing import Optional, Tuple
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
