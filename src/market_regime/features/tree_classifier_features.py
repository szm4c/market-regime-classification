import numpy as np
import pandas as pd
from market_regime.features.base import BaseFeatureBuilder


class TreeClassifierFeatureBuilder(BaseFeatureBuilder):

    def build_features(
        self, df: pd.DataFrame, is_train: bool
    ) -> tuple[np.ndarray, np.ndarray | None, pd.Index]: ...
