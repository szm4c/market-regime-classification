import numpy as np
import pandas as pd
from market_regime.features.base import BaseFeatureBuilder
from market_regime.features.utils import validate_df, calculate_log_returns


class SharpeFeatureBuilder(BaseFeatureBuilder):
    """Build log returns from price data for ARCH model."""

    def build_features(
        self,
        df: pd.DataFrame,
        is_train: bool,
    ) -> tuple[np.ndarray, np.ndarray | None, pd.Index]:
        """Return (x, y, index) for given DataFrame."""
        required_cols = ["delivery_date", "open_t", "close_t"]
        if is_train:
            required_cols.append("target")
        df = validate_df(df=df, required_cols=required_cols)
        x = (
            calculate_log_returns(df, close_col="close_t", open_col="open_t")
            .to_frame()
            .to_numpy()
        )
        y = df["target"].to_numpy() if is_train else None
        idx = df.index
        features = ["log_returns"]

        return x, y, idx, features
