from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class Segment:
    segment_id: int
    start_idx: int
    end_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect motion segments from acceleration logs.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Motion threshold above the quiet baseline for magnitude delta.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.2,
        help="Minimum segment duration in seconds.",
    )
    parser.add_argument(
        "--quiet-duration",
        type=float,
        default=0.3,
        help="Seconds below threshold required to close a segment.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=3.0,
        help="Split segments longer than this many seconds. Use 0 to disable.",
    )
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=1.0,
        help="Initial quiet window used to estimate stationary magnitude.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/segments",
        help="Directory where segment CSV and overlay PNG will be saved.",
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


def estimate_baseline(df: pd.DataFrame, baseline_seconds: float) -> float:
    baseline_window = df[df["time_sec"] <= baseline_seconds]
    if baseline_window.empty:
        baseline_window = df.head(min(len(df), 20))
    return float(baseline_window["magnitude"].mean())


def split_if_too_long(segment: Segment, df: pd.DataFrame, max_duration: float) -> list[Segment]:
    if max_duration <= 0:
        return [segment]
    start_time = float(df.loc[segment.start_idx, "time_sec"])
    end_time = float(df.loc[segment.end_idx, "time_sec"])
    if end_time - start_time <= max_duration:
        return [segment]

    pieces: list[Segment] = []
    piece_start = segment.start_idx
    while piece_start <= segment.end_idx:
        piece_start_time = float(df.loc[piece_start, "time_sec"])
        target_end_time = piece_start_time + max_duration
        candidates = df.index[(df.index >= piece_start) & (df.index <= segment.end_idx) & (df["time_sec"] <= target_end_time)]
        piece_end = int(candidates.max()) if len(candidates) else piece_start
        pieces.append(Segment(0, piece_start, piece_end))
        piece_start = piece_end + 1
    return pieces


def detect_segments(
    df: pd.DataFrame,
    threshold: float,
    min_duration: float,
    quiet_duration: float,
    baseline_seconds: float,
    max_duration: float,
) -> tuple[list[Segment], float]:
    baseline = estimate_baseline(df, baseline_seconds)
    df["magnitude_delta"] = (df["magnitude"] - baseline).abs()

    segments: list[Segment] = []
    in_motion = False
    start_idx = 0
    quiet_start_time: float | None = None

    for idx, row in df.iterrows():
        active = float(row["magnitude_delta"]) >= threshold
        current_time = float(row["time_sec"])

        if active and not in_motion:
            in_motion = True
            start_idx = int(idx)
            quiet_start_time = None
        elif active and in_motion:
            quiet_start_time = None
        elif in_motion:
            if quiet_start_time is None:
                quiet_start_time = current_time
            if current_time - quiet_start_time >= quiet_duration:
                end_idx = int(idx)
                duration = float(df.loc[end_idx, "time_sec"] - df.loc[start_idx, "time_sec"])
                if duration >= min_duration:
                    segments.append(Segment(0, start_idx, end_idx))
                in_motion = False
                quiet_start_time = None

    if in_motion:
        end_idx = int(df.index[-1])
        duration = float(df.loc[end_idx, "time_sec"] - df.loc[start_idx, "time_sec"])
        if duration >= min_duration:
            segments.append(Segment(0, start_idx, end_idx))

    split_segments: list[Segment] = []
    for segment in segments:
        split_segments.extend(split_if_too_long(segment, df, max_duration))

    numbered = [Segment(i + 1, segment.start_idx, segment.end_idx) for i, segment in enumerate(split_segments)]
    return numbered, baseline


def dominant_axis(window: pd.DataFrame) -> str:
    ranges = {
        "x": float(window["x"].max() - window["x"].min()),
        "y": float(window["y"].max() - window["y"].min()),
        "z": float(window["z"].max() - window["z"].min()),
    }
    return max(ranges, key=ranges.get)


def provisional_label(axis: str, peak_magnitude: float, baseline: float) -> str:
    if peak_magnitude - baseline >= 8.0:
        return "strong_motion"
    if axis == "x":
        return "horizontal_motion"
    if axis == "y":
        return "vertical_motion"
    return "unknown"


def segment_rows(df: pd.DataFrame, segments: list[Segment], baseline: float) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for segment in segments:
        window = df.loc[segment.start_idx : segment.end_idx]
        axis = dominant_axis(window)
        start_time = float(window["time_sec"].iloc[0])
        end_time = float(window["time_sec"].iloc[-1])
        peak_magnitude = float(window["magnitude"].max())
        rows.append(
            {
                "segment_id": segment.segment_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "peak_magnitude": peak_magnitude,
                "mean_magnitude": float(window["magnitude"].mean()),
                "dominant_axis": axis,
                "duration_feature": end_time - start_time,
                "max_x": float(window["x"].max()),
                "max_y": float(window["y"].max()),
                "max_z": float(window["z"].max()),
                "mean_x": float(window["x"].mean()),
                "mean_y": float(window["y"].mean()),
                "mean_z": float(window["z"].mean()),
                "var_x": float(np.var(window["x"], ddof=0)),
                "var_y": float(np.var(window["y"], ddof=0)),
                "var_z": float(np.var(window["z"], ddof=0)),
                "label_candidate": provisional_label(axis, peak_magnitude, baseline),
            }
        )
    return rows


def save_segments(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "segment_id",
        "start_time",
        "end_time",
        "duration",
        "peak_magnitude",
        "mean_magnitude",
        "dominant_axis",
        "duration_feature",
        "max_x",
        "max_y",
        "max_z",
        "mean_x",
        "mean_y",
        "mean_z",
        "var_x",
        "var_y",
        "var_z",
        "label_candidate",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_segments(df: pd.DataFrame, segments: list[Segment], baseline: float, threshold: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(13, 8))
    plt.plot(df["time_sec"], df["x"], label="x", linewidth=1.0, alpha=0.8)
    plt.plot(df["time_sec"], df["y"], label="y", linewidth=1.0, alpha=0.8)
    plt.plot(df["time_sec"], df["z"], label="z", linewidth=1.0, alpha=0.8)
    plt.plot(df["time_sec"], df["magnitude"], label="magnitude", linewidth=1.5, color="black")
    plt.axhline(baseline + threshold, color="red", linestyle="--", linewidth=1, label="baseline + threshold")
    plt.axhline(max(0.0, baseline - threshold), color="red", linestyle=":", linewidth=1, label="baseline - threshold")

    for segment in segments:
        start = float(df.loc[segment.start_idx, "time_sec"])
        end = float(df.loc[segment.end_idx, "time_sec"])
        plt.axvspan(start, end, color="orange", alpha=0.22)
        plt.text(start, float(df["magnitude"].max()), f"#{segment.segment_id}", va="top", fontsize=9)

    plt.xlabel("time [sec]")
    plt.ylabel("acceleration [m/s^2]")
    plt.title("Detected motion segments")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    df = load_log(input_path)
    segments, baseline = detect_segments(
        df=df,
        threshold=args.threshold,
        min_duration=args.min_duration,
        quiet_duration=args.quiet_duration,
        baseline_seconds=args.baseline_seconds,
        max_duration=args.max_duration,
    )
    rows = segment_rows(df, segments, baseline)
    csv_path = output_dir / f"{input_path.stem}_segments.csv"
    png_path = output_dir / f"{input_path.stem}_segments.png"
    save_segments(rows, csv_path)
    plot_segments(df, segments, baseline, args.threshold, png_path)
    print(f"Detected segments: {len(segments)}")
    print(f"Baseline magnitude: {baseline:.3f}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
