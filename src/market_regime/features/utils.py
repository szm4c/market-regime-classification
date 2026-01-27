"""General functions for data transform to use across many models."""

from __future__ import annotations
from typing import Sequence
import warnings
import numpy as np
import pandas as pd


def calculate_log_returns(
    df: pd.DataFrame,
    close_col: str,
    open_col: str,
) -> pd.Series:
    """
    Calculate daily log returns using open and close price columns.

    Close price is treated as a default price column, and open price is
    used for rolling contract (NaN caused by delivery date change) - in
    such situations instead of previous close (mixing of contracts), the
    open for that day is used.

    Args:
        df: DataFrame with daily index and columns "delivery_date",
            `close_col` and `open_col`.
        close_col: name of the column with close price.
        open_col: name of the column with open price.

    Returns:
        Series named "log_return_t" with daily log returns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`df` must be DataFrame, got {type(df)}.")

    # Check if required columns are present
    required_cols = ["delivery_date", close_col, open_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"`df` is missing {missing_cols} column(s).")

    # Work on sorted copy containing only required columns
    df = df[required_cols].sort_index().copy()

    # Calculate previous price
    df["price_t-1"] = (
        df.groupby("delivery_date")[close_col].shift(1).fillna(df[open_col])
    )
    # Calculate log returns
    df["log_return_t"] = np.log(df[close_col] / df["price_t-1"])

    # Check for np.nan, np.inf, -np.inf in output
    if (~np.isfinite(df["log_return_t"].to_numpy())).any():
        warnings.warn(
            "Calculated log returns contain NaNs or Inf values.", category=UserWarning
        )

    return df["log_return_t"]


def validate_df(df, required_cols: Sequence[str] | None = None) -> pd.DataFrame:
    """Validate and normalize a pandas DataFrame.

    This function normalizes `df` so that it has a sorted
    `DatetimeIndex` named "trading_date". If the current index is not
    compliant, it attempts to set the "trading_date" column as the
    index. If the final index is not a `DatetimeIndex`, it raises an
    error.

    If `required_cols` is provided, it additionally:
    - ignores "trading_date" if present in `required_cols` (since it's
      the index),
    - checks that all other required columns exist,
    - and if "delivery_date" is required, enforces dtype
      `datetime64[ns]`.

    Args:
        df: Input object expected to be a pandas DataFrame.
        required_cols: Optional sequence of required column names. If
            None, column presence checks are skipped. Defaults to None.

    Returns:
        A validated DataFrame with a sorted `DatetimeIndex` named
        "trading_date".

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
        ValueError: If "trading_date" cannot be set as a
        `DatetimeIndex`, if required columns are missing, or if
        "delivery_date" (when required) is not `datetime64[ns]`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`df` must be a pandas.DataFrame, got {type(df).__name__}.")

    out = df.copy()

    # Normalize index to DatetimeIndex named "trading_date"
    if not (
        isinstance(out.index, pd.DatetimeIndex) and out.index.name == "trading_date"
    ):
        if "trading_date" not in out.columns:
            raise ValueError(
                'Expected index to be a DatetimeIndex named "trading_date", or a '
                'column "trading_date" that can be set as such.'
            )
        out = out.set_index("trading_date", drop=True)
        out.index.name = "trading_date"

    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError(
            'Index "trading_date" must be a DatetimeIndex, got '
            f"{type(out.index).__name__}."
        )

    out = out.sort_index()

    # Required cols checks (optional)
    if required_cols is None:
        return out

    if isinstance(required_cols, (str, bytes)):
        raise TypeError(
            "`required_cols` must be a sequence of column names, not a single string."
        )

    req = [c for c in required_cols if c != "trading_date"]

    missing = [c for c in req if c not in out.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing)
            + f". Available columns: {list(out.columns)}"
        )

    if "delivery_date" in req and not pd.api.types.is_datetime64_ns_dtype(
        out["delivery_date"].dtype
    ):
        raise ValueError(
            f'Column "delivery_date" must have dtype datetime64[ns], got '
            f'{out["delivery_date"].dtype!s}.'
        )

    return out
