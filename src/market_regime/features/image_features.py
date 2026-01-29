from dataclasses import dataclass
import numpy as np
import pandas as pd
from pyts.image import RecurrencePlot, GramianAngularField
from market_regime.features.base import BaseFeatureBuilder


@dataclass
class ImageFeatureConfig:
    """Configuration for RP + GAF image featues."""

    window: int = 60
    target_col: str = "target"
    log: bool = True


class ImageFeatureBuilder(BaseFeatureBuilder):
    """Build RP + GAF image features from price data."""

    def __init__(self, config: ImageFeatureConfig | None = None):
        self.config = config or ImageFeatureConfig()
        self._rp = RecurrencePlot()
        self._gaf = GramianAngularField(method="difference")

    def build_features(
        self,
        df: pd.DataFrame,
        is_train: bool,
    ) -> tuple[np.ndarray, np.ndarray | None, pd.Index]:
        """Return (x, y, index) for given DataFrame."""
        if isinstance(df, pd.DataFrame):
            if "close_t" not in df.columns:
                raise KeyError("df must contain a 'close_t' column")
            if is_train and self.config.target_col not in df.columns:
                raise KeyError(
                    f"df must contain target column '{self.config.target_col}' when "
                    "is_train=True"
                )
        else:
            raise TypeError(f"`df` must be a DataFrame, got {type(df)}")
        # Sort index (trading_date)
        df = df.sort_index()

        # Extract price array from DataFrame
        price = df["close_t"].to_numpy()

        # Transform price into (N, W) arrays
        x_rp = transform_price(
            price,
            window=self.config.window,
            scale_method="zscore",
            log=self.config.log,
            keepdims=True,
        )
        x_gaf = transform_price(
            price,
            window=self.config.window,
            scale_method="minmax",
            log=self.config.log,
            keepdims=True,
        )

        # Use only not NaN rows
        mask = ~np.isnan(x_rp).any(axis=1) & ~np.isnan(x_gaf).any(axis=1)
        x_rp = x_rp[mask]  # (N, W)
        x_gaf = x_gaf[mask]  # (N, W)
        idx = df.index[mask]  # type: ignore

        x_rp_t = self._rp.fit_transform(x_rp)  # (N, W, W)
        x_gaf_t = self._gaf.fit_transform(x_gaf)  # (N, W, W)

        x = np.stack([x_rp_t, x_gaf_t], axis=1).astype("float32")  # (N, 2, W, W)
        y = (
            df.loc[idx, self.config.target_col].to_numpy().astype("int32")  # type: ignore
            if is_train
            else None
        )

        return x, y, idx, None


def transform_price(
    x: np.ndarray,
    window: int = 60,
    scale_method: str = "minmax",
    log: bool = True,
    keepdims: bool = False,
) -> np.ndarray:
    # Validate input
    if isinstance(scale_method, str):
        scale_method = scale_method.strip().lower().replace("-", "").replace("_", "")
    else:
        raise TypeError(f"`scale_method` must be string, got {type(scale_method)}")

    if log:
        x = np.log(x)

    x_windows_list: list = []
    for t in range(0, len(x)):
        start = t + 1 - window
        end = t + 1
        if start < 0:
            if keepdims:
                x_window = np.full((window,), np.nan)
                x_windows_list.append(x_window)
        else:
            x_window = x[start:end]
            x_windows_list.append(x_window)

    x_windows = np.vstack(x_windows_list)

    if scale_method == "minmax":
        x_min = np.min(x_windows, axis=1, keepdims=True)
        x_max = np.max(x_windows, axis=1, keepdims=True)
        x_scaled = (x_windows - x_min) / (x_max - x_min)
        x_scaled *= 2
        x_scaled -= 1
    elif scale_method == "zscore":
        x_mean = np.mean(x_windows, axis=1, keepdims=True)
        x_std = np.std(x_windows, axis=1, keepdims=True)
        x_scaled = (x_windows - x_mean) / x_std
    else:
        raise ValueError("`scale_method` must be either 'minmax' or 'zscore'.")

    return x_scaled
