from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Abstract base for estimators used in regime pipelines."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        """Fit estimator on features x and labels y."""
        ...

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels for features x."""
        ...

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities for features x."""
        ...
