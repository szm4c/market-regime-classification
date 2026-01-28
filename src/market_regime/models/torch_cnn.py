from typing import Optional, Self, Literal
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score

from market_regime.models.base import BaseModel


# Mapowanie etykiet
Y_TO_INT = {-1: 0, 0: 1, 1: 2}
INT_TO_Y = {0: -1, 1: 0, 2: 1}


def _map_y_to_int(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).ravel()
    uniq = np.unique(y)
    if not np.isin(uniq, [-1, 0, 1]).all():
        raise ValueError(f"Unexpected labels: {uniq.tolist()}")
    return np.vectorize(Y_TO_INT.get, otypes=[int])(y).astype(np.int64)


def _map_int_to_y(y_int: np.ndarray) -> np.ndarray:
    y_int = np.asarray(y_int).ravel().astype(int)
    return np.vectorize(INT_TO_Y.get, otypes=[int])(y_int).astype(int)


class SimpleCNN2Ch(nn.Module):
    def __init__(self, in_ch: int = 2, n_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


EarlyStopMetric = Literal["macro_f1", "val_loss"]


class CNNModel(BaseModel):
    """
    CNN classifier for regime labels {-1, 0, 1}.

    Expects X with shape (N, 2, W, W), where channels are [RP, GAF].
    Returns probabilities in class order [-1, 0, 1] (internally mapped
    to [0,1,2]).
    """

    def __init__(
        self,
        *,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 30,
        patience: int = 7,
        dropout: float = 0.2,
        balanced_batches: bool = True,
        early_stop_metric: EarlyStopMetric = "macro_f1",
        seed: int = 333,
        device: Optional[str] = None,
        val_frac_internal: float = 0.15,
    ) -> None:
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.dropout = float(dropout)
        self.balanced_batches = bool(balanced_batches)
        self.early_stop_metric = early_stop_metric
        self.seed = int(seed)
        self.device = device
        self.val_frac_internal = float(val_frac_internal)

        # fitted state
        self.model_: Optional[SimpleCNN2Ch] = None
        self.fitted: bool = False

    def _get_device(self) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _set_seeds(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @staticmethod
    def _ensure_4d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4 or x.shape[1] != 2 or x.shape[2] != x.shape[3]:
            raise ValueError(f"Expected x shape (N,2,W,W). Got {x.shape}")
        return x

    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        x = self._ensure_4d(x)
        y_int = _map_y_to_int(y)

        if x.shape[0] != y_int.shape[0]:
            raise ValueError("X and y have different number of rows.")

        self._set_seeds()
        device = self._get_device()

        X = torch.from_numpy(x).to(device)
        Y = torch.from_numpy(y_int).to(device)

        # internal temporal split (last val_frac of training window as val)
        n = X.shape[0]
        n_val = int(round(n * self.val_frac_internal))
        n_val = max(1, min(n - 1, n_val))

        Xtr, Ytr = X[:-n_val], Y[:-n_val]
        Xva, Yva = X[-n_val:], Y[-n_val:]

        model = SimpleCNN2Ch(in_ch=2, n_classes=3, dropout=self.dropout).to(device)
        opt = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        bs = max(8, min(self.batch_size, Xtr.shape[0]))

        # dataloader
        ds = TensorDataset(Xtr, Ytr)
        if self.balanced_batches:
            counts = torch.bincount(Ytr, minlength=3).float()
            class_w = (counts.sum() / (3.0 * counts)).detach().cpu()
            sample_w = class_w[Ytr.detach().cpu()]
            sampler = WeightedRandomSampler(
                sample_w, num_samples=len(sample_w), replacement=True
            )
            loader = DataLoader(ds, batch_size=bs, sampler=sampler)
        else:
            loader = DataLoader(ds, batch_size=bs, shuffle=True)

        # early stopping config
        if self.early_stop_metric == "val_loss":
            best_score = float("inf")

            def better(new: float, best: float) -> bool:
                return (best - new) > 1e-4

        elif self.early_stop_metric == "macro_f1":
            best_score = -float("inf")

            def better(new: float, best: float) -> bool:
                return (new - best) > 1e-4

        else:
            raise ValueError(f"Unknown early_stop_metric={self.early_stop_metric}")

        best_state = None
        bad_epochs = 0

        for _epoch in range(1, self.epochs + 1):
            # train
            model.train()
            for xb, yb in loader:
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

            # validate
            model.eval()
            with torch.no_grad():
                val_logits = model(Xva)
                val_loss = float(loss_fn(val_logits, Yva).item())

                pred_int = torch.argmax(val_logits, dim=1).detach().cpu().numpy()
                y_true = _map_int_to_y(Yva.detach().cpu().numpy())
                y_pred = _map_int_to_y(pred_int)
                val_f1 = float(f1_score(y_true, y_pred, average="macro"))

            score = val_loss if self.early_stop_metric == "val_loss" else val_f1

            if better(score, best_score):
                best_score = score
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        self.model_ = model
        self.fitted = True
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted or self.model_ is None:
            raise RuntimeError("Model must be fitted before predicting.")

        x = self._ensure_4d(x)
        device = self._get_device()

        self.model_.eval()
        with torch.no_grad():
            X = torch.from_numpy(x).to(device)
            logits = self.model_(X)
            proba = torch.softmax(logits, dim=1).detach().cpu().numpy()

        if proba.ndim != 2 or proba.shape[1] != 3:
            raise RuntimeError(f"Expected proba shape (n,3). Got {proba.shape}")

        return proba  # order corresponds to int classes 0,1,2 => labels -1,0,1

    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)
        pred_int = np.argmax(proba, axis=1).astype(int)
        return _map_int_to_y(pred_int)
