from pathlib import Path
import json
import pandas as pd


def get_data_path(
    repo_name: str = "market-regime-classification",
    data_relpath: str | Path = "data/preprocessed",
) -> Path:
    """
    Return the absolute path to the repo's preprocessed data directory,
    regardless of where this function is called from inside the repo.

    The function walks upward from `start` (or this file's location, if
    available; otherwise the current working directory) until it finds a
    directory whose name matches `repo_name`.

    Args:
        repo_name: Name of the repository root directory.
        data_relpath: Relative path from repo root to the data directory.

    Returns:
        Absolute Path to the data directory.
    """
    data_relpath = Path(data_relpath)

    # Prefer current location of the code, fall back to current working dir.
    try:
        start_path = Path(__file__).resolve().parent
    except NameError:
        start_path = Path.cwd().resolve()

    # If `start_path` is a file, search from its parent.
    if start_path.is_file():
        start_path = start_path.parent

    for parent in [start_path, *start_path.parents]:
        if parent.name == repo_name:
            return (parent / data_relpath).resolve()

    raise RuntimeError(
        f"Could not locate repo root named '{repo_name}' by searching upward "
        f"from '{start_path}'."
    )


def get_data(preprocessed: bool = True):
    """
    Load data with daily prices.

    Args:
        preprocessed: If true returns preprocessed data with labels,
            otherwise raw data is returned. Defaults to True.

    Returns:
        DataFrame with daily prices.
    """
    # Find file path
    data_path = get_data_path()
    file_name = "daily_prices_with_labels.csv" if preprocessed else "daily_prices.csv"
    file_path = data_path / file_name

    # Load data
    df = pd.read_csv(file_path).reset_index(drop=False)

    # Check required columns
    required_cols = ["trading_date", "delivery_date", "open", "close"]
    if preprocessed:
        required_cols.append("target")
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"data is missing required columns: {missing_cols}.")

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

    return df


def save_params(params: dict, name: str) -> None:
    file_name = name.rstrip(".json")
    data_path = get_data_path(data_relpath="data/params")
    file_path = data_path / file_name

    def _default(o):
        # np.float64, np.int64 itp.
        try:
            import numpy as np

            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
        except Exception:
            pass
        # fallback
        return str(o)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False, default=_default)


def load_params(name: str) -> dict:
    file_name = name if name.endswith(".json") else f"{name}.json"
    data_path = get_data_path(data_relpath="data/params")
    file_path = data_path / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"Params file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        params = json.load(f)

    if not isinstance(params, dict):
        raise ValueError(f"Expected dict in {file_path}, got {type(params)}")

    return params
