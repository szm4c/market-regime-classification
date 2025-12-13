"""General functions for data transform to use across many models."""

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
