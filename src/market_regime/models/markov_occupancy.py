from typing import Self
import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from market_regime.models.base import BaseModel


class MarkovOccupancyModel(BaseModel):
    """
    Clean baseline Markov-switching regime classifier.

    Steps:
    1) Fit MarkovRegression on returns with k_regimes=3.
    2) Map regimes to {-1,0,1} using sorted regime means (low->-1, mid->0, high->1).
    3) Take last filtered regime probs pi_t.
    4) Forecast pi_{t+h} for h=1..window_len using transition matrix P.
    5) Compute occupancy scores: score_k = sum_h pi_{t+h}(k)
    6) Predict label of argmax score_k.
    """

    def __init__(
        self,
        window_len: int = 20,
        k_regimes: int = 3,
        trend: str = "c",
        switching_variance: bool = True,
        maxiter: int = 2000,
    ):
        if k_regimes != 3:
            raise ValueError("This baseline assumes k_regimes=3 to map to {-1,0,1}.")

        self.window_len = window_len
        self.k_regimes = k_regimes
        self.trend = trend
        self.switching_variance = switching_variance
        self.maxiter = maxiter
        self.fitted = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        r = np.asarray(x).ravel()

        mod = MarkovRegression(
            r,
            k_regimes=self.k_regimes,
            trend=self.trend,
            switching_variance=self.switching_variance,
        )
        self.res_ = mod.fit(disp=False, maxiter=self.maxiter)
        self.fitted = True

        # Transition matrix P (k x k)
        self.P_ = np.squeeze(np.asarray(self.res_.regime_transition))
        if self.P_.shape != (self.k_regimes, self.k_regimes):
            raise RuntimeError(f"Unexpected transition matrix shape: {self.P_.shape}")

        # Filtered probabilities at last point pi_t (k,)
        probs = self.res_.filtered_marginal_probabilities
        if hasattr(probs, "iloc"):
            self.pi_t_ = probs.iloc[-1].to_numpy()
        else:
            self.pi_t_ = np.asarray(probs)[-1]

        # Extract per-regime means (trend="c" -> const[k] params)
        self.mu_ = self._extract_regime_means()

        # Map regime index -> label (-1,0,1) by ordering mu
        order = np.argsort(self.mu_)
        self.state_to_label_ = {order[0]: -1, order[1]: 0, order[2]: 1}

        return self

    def _extract_regime_means(self) -> np.ndarray:
        """
        Extract per-regime means from fitted MarkovRegression.

        Works for trend="c" where param names include const[0], const[1], const[2].
        """
        names = self.res_.model.param_names
        params = self.res_.params

        mu = np.zeros(self.k_regimes)
        for k in range(self.k_regimes):
            key = f"const[{k}]"
            if key not in names:
                raise RuntimeError(
                    f"Could not find {key} in param names. " f"Available names: {names}"
                )
            mu[k] = params[names.index(key)]
        return mu

    def _forecast_occupancy_scores(self) -> np.ndarray:
        """
        Forecast occupancy scores over next window_len days.

        Returns scores shape (k_regimes,) where score_k = sum_{h=1..H} pi_{t+h}(k).
        """
        pi = self.pi_t_.copy().astype(float)
        scores = np.zeros(self.k_regimes, dtype=float)

        for _ in range(self.window_len):
            pi = pi.dot(self.P_)  # guaranteed (k,)
            scores += pi

        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels for x.

        Note:
        - In your walk-forward setup you usually predict one row at a time.
        - We use the fitted model (trained up to time t) and produce a label for each row.
        - For batch x (n>1), this baseline returns the same label for all rows, because
          it's a pure "from end-of-train" forecast baseline.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        scores = self._forecast_occupancy_scores()
        best_state = int(np.argmax(scores))
        label = self.state_to_label_[best_state]

        n = x.shape[0]
        return np.full(n, label, dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This model does not predict probabilities.")
