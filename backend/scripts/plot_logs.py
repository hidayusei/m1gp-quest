from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot acceleration CSV logs.")
    parser.add_argument("--input", required=True, help="Input CSV path under data/raw or any CSV path.")
    parser.add_argument(
        "--output-dir",
        default="data/processed/plots",
        help="Directory where PNG plots will be saved.",
    )
    return parser.parse_args()


def load_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "x", "y", "z", "magnitude"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["time_sec"] = (df["timestamp"] - df["timestamp"].iloc[0]) / 1000.0
    return df


def plot_log(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    for column in ["x", "y", "z", "magnitude"]:
        plt.plot(df["time_sec"], df[column], label=column, linewidth=1.2)
    plt.xlabel("time [sec]")
    plt.ylabel("acceleration [m/s^2]")
    plt.title("Acceleration log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    df = load_log(input_path)
    output_path = output_dir / f"{input_path.stem}_plot.png"
    plot_log(df, output_path)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
