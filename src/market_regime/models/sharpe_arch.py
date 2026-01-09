from typing import Self
from dataclasses import dataclass
import warnings
import numpy as np
from arch import arch_model
from market_regime.models.base import BaseModel
from market_regime.labels import calculate_best_k


class SharpeModel(BaseModel):

    def __init__(
        self,
        window_len: int = 20,
        mean="AR",
        lags=1,
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist="t",
        k=None,
    ):
        self.window_len = window_len
        self.mean = mean
        self.lags = lags
        self.vol = vol
        self.p = p
        self.o = o
        self.q = q
        self.dist = dist
        self.k = k
        self.fitted = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        """Fit estimator on features x and labels y."""
        # Scale `x` that is assumed to be log-returns to value in %
        lr_pct = x * 100

        # Fit model and store result object (fitted model)
        self.model_res = arch_model(
            y=lr_pct,
            x=None,
            mean=self.mean,
            lags=self.lags,
            vol=self.vol,
            p=self.p,
            o=self.o,
            q=self.q,
            dist=self.dist,
        ).fit(disp="off")
        self.fitted = True

        if self.k is None:
            # In-sample forecast in order to estimate `k` parameter
            start = self.model_res.model._fit_indices[0]
            # for each point in train data predict a 20 steps ahead forecast that will be
            # used to create sharpe ratio for 20 day ahead window that is in fact a forecast
            # from a single day
            in_sample_fc = self.model_res.forecast(horizon=20, start=start)
            lr_pct_pred = in_sample_fc.mean
            lr_pred = lr_pct_pred / 100
            lr_pred = lr_pred.sum(axis=1).to_numpy()
            lr_pct_var_pred = in_sample_fc.variance
            lr_var_pred = lr_pct_var_pred / 100**2
            lr_var_pred = lr_var_pred.sum(axis=1).to_numpy()
            lr_vol_pred = np.sqrt(lr_var_pred)
            sharpe_pred = lr_pred / lr_vol_pred
            # Calculate `k` parameter using in-sample forecast on train data
            self.k = calculate_best_k(sharpe_pred)

        return self

    def predict_sharpe(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return Exception("In order to make a prediction model must be fitted.")
        pred_len = x.shape[0]
        fc_horizon = pred_len + self.window_len - 1
        fc = self.model_res.forecast(horizon=fc_horizon)
        lr_pct_pred = fc.mean
        lr_pred = lr_pct_pred / 100
        lr_pred = (
            lr_pred.T.reset_index(drop=True)
            .iloc[:, 0]
            .rolling(window=20, min_periods=20)
            .sum()
            .dropna()
            .to_numpy()
        )
        lr_pct_var_pred = fc.variance
        lr_var_pred = lr_pct_var_pred / 100**2
        lr_var_pred = (
            lr_var_pred.T.reset_index(drop=True)
            .iloc[:, 0]
            .rolling(window=20, min_periods=20)
            .sum()
            .dropna()
            .to_numpy()
        )
        lr_vol_pred = np.sqrt(lr_var_pred)

        return lr_pred / lr_vol_pred

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels `x.shape[0]` steps ahead."""
        sharpe_pred = self.predict_sharpe(x=x)

        return np.where(
            sharpe_pred <= -self.k, -1, np.where(sharpe_pred >= self.k, 1, 0)
        )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities for features x."""
        raise NotImplementedError("This model does not predict probabilities.")
