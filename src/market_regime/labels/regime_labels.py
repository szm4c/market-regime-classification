import numpy as np
import pandas as pd
from market_regime.features import calculate_log_returns


def calculate_best_k(a: np.ndarray) -> float:
    """
    Compute a symmetric threshold k that balances three classes.

    The array `a` is treated as a score (e.g. future Sharpe-like ratio).
    For each candidate k, points are split into:
        - bearish: a <= -k
        - neutral: -k < a < k
        - bullish: a >= k
    The function chooses k that makes the three class shares as close
    as possible to 1/3 each. If no better k is found, 0.0 is returned.

    Args:
        a: One-dimensional array of real-valued scores.

    Returns:
        The selected symmetric threshold k.
    """
    ks = np.linspace(0, np.abs(a).max(), 500)
    best_k = 0.0
    best_score = np.inf

    for k in ks:
        bearish = np.mean(a <= -k)
        bullish = np.mean(a >= k)
        neutral = 1 - bearish - bullish
        probs = np.array([bearish, neutral, bullish])
        score = ((probs - 1 / 3) ** 2).sum()  # how far from perfect 1/3-1/3-1/3

        if score < best_score:
            best_score = score
            best_k = k

    return best_k


def make_regime_labels(
    daily_prices: pd.DataFrame,
    window_len: int = 20,
    delivery_date_cutoff: pd.Timestamp = pd.Timestamp("2025-01-01"),
) -> tuple[pd.Series, float]:
    """
    Create 3-class regime labels from raw daily prices.

    The label at trading date t is based on the Sharpe-like ratio of the
    log return over the next `window_len` days divided by the volatility
    (std of daily log returns * sqrt(window_len)) over the same future
    window. A symmetric threshold ±k is chosen to make class shares
    (bearish, neutral, bullish) as close as possible to 1/3 each.

    Args:
        daily_prices: DataFrame with columns ['trading_date',
            'delivery_date', 'open', 'close']. Function expects one row
            per trading_date for the relevant contract.
        window_len: Horizon in business days for the future window.
        delivery_date_cutoff: Rows with delivery_date >= cutoff are
            dropped after label construction.

    Returns:
        A tuple (target, k) where:
            target: Series of labels {-1, 0, 1} indexed by trading_date.
            k: Symmetric threshold used to define regimes.
    """
    # Validate `daily_prices`
    if not isinstance(daily_prices, pd.DataFrame):
        raise TypeError(
            f"`daily_prices` must be a pandas DataFrame. Got {type(daily_prices)}."
        )

    df = daily_prices.copy().reset_index(drop=False)

    # Check required columns
    required_cols: list[str] = ["trading_date", "delivery_date", "open", "close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"`daily_prices` is missing required columns: {missing_cols}.")

    # Ensure datetime dtypes
    df["trading_date"] = pd.to_datetime(df["trading_date"], format="%Y-%m-%d")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], format="%Y-%m-%d")

    # Rename price columns, set index, sort by trading date
    df = (
        df[required_cols]
        .rename(columns={"open": "open_t", "close": "close_t"})
        .set_index("trading_date")
        .sort_index()
    )

    # Validate `window_len`
    try:
        window_len = int(window_len)
    except Exception as e:
        raise TypeError("`window_len` must be an integer.") from e
    if window_len < 1:
        raise ValueError("`window_len` must be at least 1.")

    # Validate `delivery_date_cutoff`
    if not isinstance(delivery_date_cutoff, pd.Timestamp):
        raise TypeError("`delivery_date_cutoff` must be a pandas Timestamp.")
    if delivery_date_cutoff <= df["delivery_date"].min():
        raise ValueError(
            "`delivery_date_cutoff` must be greater than the minimum delivery_date in "
            f"data ({df['delivery_date'].min()})."
        )

    # Calculate log returns
    df["log_return_t"] = calculate_log_returns(
        df=df, close_col="close_t", open_col="open_t"
    )
    df[f"log_return_t-{window_len-1}:t"] = (
        df["log_return_t"]
        .rolling(window=window_len, min_periods=window_len, center=False)
        .sum()
    )
    df[f"log_return_t+1:t+{window_len}"] = df[f"log_return_t-{window_len-1}:t"].shift(
        -window_len
    )

    # Calculate volatility
    df[f"volatility_t-{window_len-1}:t"] = df["log_return_t"].rolling(
        window=window_len, min_periods=window_len, center=False
    ).std() * np.sqrt(window_len)
    df[f"volatility_t+1:t+{window_len}"] = df[f"volatility_t-{window_len-1}:t"].shift(
        -window_len
    )

    # Calculate sharpe ratio (mean / std)
    df[f"sharpe_ratio_{window_len}"] = (
        df[f"log_return_t+1:t+{window_len}"] / df[f"volatility_t+1:t+{window_len}"]
    )

    df = df[df["delivery_date"] < delivery_date_cutoff]

    # Find symmetrical threshold that will maximize symmetry of target labels
    sharpe_ratio_no_nan = df[f"sharpe_ratio_{window_len}"].dropna().to_numpy()
    if sharpe_ratio_no_nan.size == 0:
        raise ValueError("Not enough data to compute regime labels (all Sharpe NaN).")

    k = round(calculate_best_k(sharpe_ratio_no_nan), 2)

    # Create regime labels: -1 (bearish), 0 (no trend), 1 (bullish)
    df["target"] = 0
    df.loc[df[f"sharpe_ratio_{window_len}"] >= k, "target"] = 1
    df.loc[df[f"sharpe_ratio_{window_len}"] <= -k, "target"] = -1

    # Return only dates where Sharpe is defined (full future window)
    target = df.loc[df[f"sharpe_ratio_{window_len}"].notna(), "target"].copy()

    return target, k
