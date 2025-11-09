from pathlib import Path
import pandas as pd
from market_regime.labeling import make_regime_labels


def main() -> None:
    """Create regime labels and save daily prices with labels to data/preprocessed."""
    # Resolve repo root as parent of this script's directory
    repo_dir = Path(__file__).resolve().parents[1]
    if repo_dir.name != "market-regime-classification":
        raise RuntimeError(
            f"Expected repo dir 'market-regime-classification', got {repo_dir.name!r}."
        )

    raw_data_path = repo_dir / "data" / "raw"
    preprocessed_data_path = repo_dir / "data" / "preprocessed"
    preprocessed_data_path.mkdir(parents=True, exist_ok=True)

    input_path = raw_data_path / "daily_prices.csv"
    output_path = preprocessed_data_path / "daily_prices_with_labels.csv"

    # Load raw daily prices
    daily_prices = pd.read_csv(input_path)

    # Create labels (y) and threshold k
    target, k = make_regime_labels(
        daily_prices=daily_prices,
        window_len=20,
        delivery_date_cutoff=pd.Timestamp("2025-01-01"),
    )

    # Join labels back to original data, indexed by trading_date
    df = daily_prices.copy()
    df = df.set_index("trading_date")
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d")

    # Remove any previous 'target' column just in case
    df = df.drop(columns="target", errors="ignore")
    df = df.join(target.rename("target"))

    # Get rid of NaN tail from target, reset index for saving to CSV
    df = df.dropna().reset_index()

    # Save to preprocessed path
    df.to_csv(output_path, index=False)

    print(f"Saved labeled daily prices to: {output_path}")
    print(f"Used symmetric threshold k = {k:.2f}")


if __name__ == "__main__":
    main()
