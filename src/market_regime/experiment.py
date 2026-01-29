import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)
from market_regime.pipelines.base import RegimePipeline

# labels order
LABELS = [-1, 0, 1]
LABEL_TO_INT = {-1: 0, 0: 1, 1: 2}
INT_TO_LABEL = {0: -1, 1: 0, 2: 1}


def evaluate_predictions(df: pd.DataFrame, pred: pd.Series, name: str) -> dict:
    """Evaluate a prediction series aligned to df['target']."""
    pred = pred.dropna().astype(int)
    y_true = df.loc[pred.index, "target"].astype(int)
    y_pred = pred.astype(int)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(
        cm, index=[f"true_{c}" for c in LABELS], columns=[f"pred_{c}" for c in LABELS]
    )

    out = {
        "name": name,
        "n": int(len(pred)),
        "macro_f1": float(macro_f1),
        "balanced_acc": float(bal_acc),
        "cm": cm_df,
        "y_true_dist": y_true.value_counts(normalize=True).reindex(LABELS).fillna(0.0),
        "y_pred_dist": y_pred.value_counts(normalize=True).reindex(LABELS).fillna(0.0),
        "report": classification_report(y_true, y_pred, labels=LABELS, zero_division=0),
    }
    return out


def print_eval(eval_out: dict) -> None:
    print(f"\n=== {eval_out['name']} ===")
    print(f"n = {eval_out['n']}")
    print(f"macro_f1        = {eval_out['macro_f1']:.4f}")
    print(f"balanced_acc    = {eval_out['balanced_acc']:.4f}\n")
    print("Confusion matrix:")
    print(eval_out["cm"])
    print("\nClass distribution:")
    print("y_true:\n", eval_out["y_true_dist"])
    print("\ny_pred:\n", eval_out["y_pred_dist"])
    print("\nClassification report:\n", eval_out["report"])


def baseline_prev_label(
    df: pd.DataFrame,
    oos_start: str | pd.Timestamp,
    init_train_end: str | pd.Timestamp,
    label_horizon: int = 20,
) -> pd.Series:
    """
    Predict y_d using the last label that would be known at time d
    (y_{d-(H+1)} where H = label_horizon).

    This matches the same information constraint as OOS loops with
    train_end = d-(H+1).
    """
    oos_start = pd.Timestamp(oos_start)
    init_train_end = pd.Timestamp(init_train_end)

    # last known label at day d is target shifted by (H+1) trading days
    pred = df["target"].shift(label_horizon + 1)

    oos_idx = df.index[df.index >= oos_start]
    if len(oos_idx) == 0:
        raise ValueError("No OOS dates after oos_start.")

    out = pred.loc[oos_idx].astype("float64").rename("prediction")

    # fallback
    if out.notna().sum() == 0:
        last_lab = float(int(df.loc[init_train_end, "target"]))
        out[:] = last_lab
    else:
        if pd.isna(out.iloc[0]):
            out.iloc[0] = float(int(df.loc[init_train_end, "target"]))

    return out


def baseline_freq_sampler(
    df: pd.DataFrame,
    oos_start: str | pd.Timestamp,
    init_train_end: str | pd.Timestamp,
    label_horizon: int = 20,
    seed: int = 333,
    smoothing: float = 1.0,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    oos_start = pd.Timestamp(oos_start)
    init_train_end = pd.Timestamp(init_train_end)

    oos_idx = df.index[df.index >= oos_start]
    if len(oos_idx) == 0:
        raise ValueError("No OOS dates after oos_start.")

    # W dniu oos_start znamy etykiety maksymalnie do oos_start-(H+1)
    oos_pos0 = df.index.get_loc(oos_idx[0])
    known_cutoff_pos = oos_pos0 - (label_horizon + 1)
    if known_cutoff_pos < 0:
        raise ValueError("Not enough history before oos_start for given label_horizon.")

    init_end_pos = df.index.get_loc(init_train_end)
    seed_end_pos = min(init_end_pos, known_cutoff_pos)

    init_y = df.iloc[: seed_end_pos + 1]["target"].astype(int).to_numpy()
    counts = np.zeros(3, dtype=np.float64)
    for lab in init_y:
        counts[LABEL_TO_INT[int(lab)]] += 1.0
    counts = counts + float(smoothing)

    preds = []
    for d in oos_idx:
        probs = counts / counts.sum()
        pred_int = int(rng.choice([0, 1, 2], p=probs))
        preds.append((d, INT_TO_LABEL[pred_int]))

        # Po dniu d dochodzi etykieta z indeksu (pos(d) - H)
        pos = df.index.get_loc(d)
        upd_pos = pos - label_horizon
        if upd_pos >= 0:
            lab = df.iloc[upd_pos]["target"]
            if pd.notna(lab):
                counts[LABEL_TO_INT[int(lab)]] += 1.0

    return pd.Series(
        [v for _, v in preds],
        index=[k for k, _ in preds],
        name="prediction",
        dtype="float64",
    )


def oos_predict_with_tail(
    df: pd.DataFrame,
    rp: RegimePipeline,
    oos_start: str | pd.Timestamp,
    min_history_days: int = 0,
    label_horizon: int = 20,
    desc: str = "OOS Prediction",
) -> pd.Series:
    """
    For each day d>=oos_start:
      - train on df[:train_end] where train_end = d - (label_horizon+1)
      - test on df[:d] to allow feature builder to use full history/tail
      - extract prediction for day d if available, else NaN
    """
    oos_start = pd.Timestamp(oos_start)
    oos_days = df.index[df.index >= oos_start]

    preds = []
    for d in tqdm(oos_days, desc=desc):
        pos = df.index.get_loc(d)
        if pos < max(min_history_days, label_horizon + 1):
            continue

        train_end = df.index[pos - (label_horizon + 1)]
        df_train = df.loc[:train_end, :]
        df_test = df.loc[:d, :]

        rp.fit(df_train)
        full_pred = rp.predict(df_test)

        if d in full_pred.index:
            pred = full_pred.loc[[d]]
        else:
            pred = pd.Series([np.nan], index=[d], name="prediction")

        preds.append(pred)

    if len(preds) == 0:
        return pd.Series([], dtype="float64", name="prediction")

    return pd.concat(preds).sort_index()


def oos_predict_one_row(
    df: pd.DataFrame,
    rp: RegimePipeline,
    oos_start: str | pd.Timestamp,
    desc: str = "OOS Prediction",
) -> pd.Series:
    oos_start = pd.Timestamp(oos_start)
    oos_days = df.index[df.index >= oos_start]

    preds = []
    for d in tqdm(oos_days, desc=desc):
        df_train = df.loc[: d - pd.Timedelta(days=1), :]
        df_test = df.loc[[d], :]
        rp.fit(df_train)
        pred = rp.predict(df_test)
        preds.append(pred)

    if len(preds) == 0:
        return pd.Series([], dtype="float64", name="prediction")

    return pd.concat(preds).sort_index()
