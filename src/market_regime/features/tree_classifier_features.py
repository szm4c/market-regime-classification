from warnings import catch_warnings, filterwarnings
from itertools import combinations
import numpy as np
import pandas as pd
from market_regime.features.base import BaseFeatureBuilder
from market_regime.features.utils import validate_df, calculate_log_returns


class TreeClassifierFeatureBuilder(BaseFeatureBuilder):

    def build_features(
        self, df: pd.DataFrame, is_train: bool
    ) -> tuple[np.ndarray, np.ndarray | None, pd.Index]:
        """Return (x, y, index) for given DataFrame."""
        # Validate input
        required_cols = ["delivery_date", "open_t", "close_t"]
        if is_train:
            required_cols.append("target")
        df = validate_df(df=df, required_cols=required_cols)

        with catch_warnings():
            filterwarnings("ignore")
            # Features derived from DatetimeIndex and 'delivery_date' column
            df["contract_change_flag"] = (
                df["delivery_date"] != df["delivery_date"].shift(1)
            ).astype(int)
            df["contract_quarter"] = df["delivery_date"].dt.quarter.astype(int)
            df["contract_month"] = df["delivery_date"].dt.month.astype(int)
            df["day_of_week"] = df.index.day_of_week.astype(int)

            # Price lags
            df["close_t-1"] = (
                df.groupby("delivery_date")["close_t"].shift(1).fillna(df["open_t"])
            )

            # Calculate log returns
            df["log_returns_t"] = calculate_log_returns(
                df, close_col="close_t", open_col="open_t"
            )
            df["lr2"] = df["log_returns_t"] ** 2

            # Log returns lags
            for lag in range(1, 6):
                df[f"lr_lag_{lag}"] = df["log_returns_t"].shift(lag)
                df[f"lr2_lag_{lag}"] = df["lr2"].shift(lag)

            # Price move between two sessions (gap)
            df["gap"] = df["open_t"] - df["close_t-1"]
            df["lr_gap"] = np.log(df["open_t"] / df["close_t-1"])

            # Intraday price moves
            df["price_move_id"] = df["close_t"] - df["open_t"]
            df["price_move_id_abs"] = df["price_move_id"].abs()
            df["lr_id"] = np.log(df["close_t"] / df["open_t"])
            df["lr2_id"] = df["lr_id"] ** 2

            # Rolling features
            windows = [5, 10, 20, 40, 60]
            for window in windows:
                # price move in window (lr is additive)
                df[f"lr_sum_{window}"] = (
                    df["log_returns_t"].rolling(window=window, min_periods=window).sum()
                )
                # volatility in window
                df[f"lr_std_{window}"] = (
                    df["log_returns_t"].rolling(window=window, min_periods=window).std()
                )
                # lags of lr price move and volatility (lr std)
                for lag in range(1, 4):
                    df[f"lr_sum_{window}_lag_{lag*window}"] = df[
                        f"lr_sum_{window}"
                    ].shift(lag * window)
                    df[f"lr_std_{window}_lag_{lag*window}"] = df[
                        f"lr_std_{window}"
                    ].shift(lag * window)
                # moving average (on close)
                # df[f"close_ma_{window}"] = (
                #     df["close_t"].rolling(window=window, min_periods=window).mean()
                # )
                df[f"close_ewma_{window}"] = (
                    df["close_t"].ewm(span=window, min_periods=window).mean()
                )
                df[f"close_min_{window}"] = (
                    df["close_t"].rolling(window=window, min_periods=window).min()
                )
                df[f"close_max_{window}"] = (
                    df["close_t"].rolling(window=window, min_periods=window).max()
                )
                # df[f"close_minus_ma_{window}"] = (
                #     df["close_t"] - df[f"close_ma_{window}"]
                # )
                df[f"close_minus_ewma_{window}"] = (
                    df["close_t"] - df[f"close_ewma_{window}"]
                )
                df[f"close_range_pos_{window}"] = (
                    df["close_t"] - df[f"close_min_{window}"]
                ) / (df[f"close_max_{window}"] - df[f"close_min_{window}"])
                df[f"close_to_max_{window}"] = (
                    df["close_t"] / df[f"close_max_{window}"] - 1.0
                )
                df[f"close_to_min_{window}"] = (
                    df["close_t"] / df[f"close_min_{window}"] - 1.0
                )

            # for window1, window2 in combinations(windows, r=2):
            #     df[f"close_ma_{window1}_vs_{window2}"] = (
            #         df[f"close_ma_{window1}"] / df[f"close_ma_{window2}"]
            #     )
            #     df[f"close_ewma_{window1}_vs_{window2}"] = (
            #         df[f"close_ewma_{window1}"] / df[f"close_ewma_{window2}"]
            #     )
            #     df[f"lr_std_{window1}_vs_{window2}"] = (
            #         df[f"lr_std_{window1}"] / df[f"lr_std_{window2}"]
            #     )

            df = df.dropna()
            x = df.drop(columns=["delivery_date", "target"], errors="ignore").to_numpy()
            y = df["target"].to_numpy() if is_train else None
            idx = df.index

        return x, y, idx
