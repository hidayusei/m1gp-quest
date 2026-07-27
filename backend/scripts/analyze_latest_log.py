from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
MIN_DURATION = 0.2
QUIET_DURATION = 0.3
BASELINE_SECONDS = 1.0
MAX_DURATION = 3.0


@dataclass
class Segment:
    segment_id: int
    start_idx: int
    end_idx: int


def backend_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def find_latest_csv(raw_dir: Path) -> Path:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    csv_files = [path for path in raw_dir.glob("*.csv") if path.is_file()]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {raw_dir}")

    return max(csv_files, key=lambda path: path.stat().st_mtime)


def normalize_time_seconds(timestamp: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(timestamp, errors="coerce")
    if numeric.isna().any():
        raise ValueError("timestamp column contains non-numeric values")

    values = numeric.to_numpy(dtype=float)
    if len(values) < 2:
        return pd.Series(np.zeros(len(values)), index=timestamp.index)

    diffs = np.diff(np.sort(values))
    positive_diffs = diffs[diffs > 0]
    median_diff = float(np.median(positive_diffs)) if len(positive_diffs) else 0.0

    # Date.now() logs are milliseconds. UNIX seconds and relative seconds usually
    # have sub-second diffs. Very large diffs are treated as microseconds.
    if median_diff > 10_000:
        scale = 1_000_000.0
    elif median_diff > 10:
        scale = 1_000.0
    else:
        scale = 1.0

    time_sec = (numeric - numeric.iloc[0]) / scale
    if float(time_sec.max() - time_sec.min()) > 24 * 60 * 60 and scale == 1.0:
        time_sec = (numeric - numeric.iloc[0]) / 1_000.0
    return time_sec


def load_log(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV: {path} ({exc})") from exc

    required = {"timestamp", "x", "y", "z"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in ["timestamp", "x", "y", "z"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    bad_rows = df[["timestamp", "x", "y", "z"]].isna().any(axis=1)
    if bad_rows.any():
        removed = int(bad_rows.sum())
        df = df.loc[~bad_rows].copy()
        print(f"Warning: removed {removed} rows with invalid numeric values.")

    if df.empty:
        raise ValueError("CSV has no valid rows after numeric cleanup")

    if "magnitude" not in df.columns:
        df["magnitude"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    else:
        df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
        missing_magnitude = df["magnitude"].isna()
        if missing_magnitude.any():
            df.loc[missing_magnitude, "magnitude"] = np.sqrt(
                df.loc[missing_magnitude, "x"] ** 2
                + df.loc[missing_magnitude, "y"] ** 2
                + df.loc[missing_magnitude, "z"] ** 2
            )

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["time_sec"] = normalize_time_seconds(df["timestamp"])
    return df


def plot_raw_waveform(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(13, 7))
    for column in ["x", "y", "z", "magnitude"]:
        plt.plot(df["time_sec"], df[column], label=column, linewidth=1.1)
    plt.xlabel("time [sec]")
    plt.ylabel("acceleration")
    plt.title("Raw acceleration waveform")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_magnitude_waveform(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(13, 5))
    plt.plot(df["time_sec"], df["magnitude"], label="magnitude", color="black", linewidth=1.4)
    plt.xlabel("time [sec]")
    plt.ylabel("magnitude")
    plt.title("Magnitude waveform")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def estimate_baseline(df: pd.DataFrame) -> float:
    baseline_window = df[df["time_sec"] <= BASELINE_SECONDS]
    if baseline_window.empty:
        baseline_window = df.head(min(len(df), 20))
    return float(baseline_window["magnitude"].mean())


def split_if_too_long(segment: Segment, df: pd.DataFrame) -> list[Segment]:
    if MAX_DURATION <= 0:
        return [segment]

    start_time = float(df.loc[segment.start_idx, "time_sec"])
    end_time = float(df.loc[segment.end_idx, "time_sec"])
    if end_time - start_time <= MAX_DURATION:
        return [segment]

    pieces: list[Segment] = []
    piece_start = segment.start_idx
    while piece_start <= segment.end_idx:
        piece_start_time = float(df.loc[piece_start, "time_sec"])
        target_end_time = piece_start_time + MAX_DURATION
        candidates = df.index[
            (df.index >= piece_start)
            & (df.index <= segment.end_idx)
            & (df["time_sec"] <= target_end_time)
        ]
        piece_end = int(candidates.max()) if len(candidates) else piece_start
        pieces.append(Segment(0, piece_start, piece_end))
        piece_start = piece_end + 1
    return pieces


def detect_segments(df: pd.DataFrame, threshold: float) -> tuple[list[Segment], float]:
    baseline = estimate_baseline(df)
    work = df.copy()
    work["magnitude_delta"] = (work["magnitude"] - baseline).abs()

    segments: list[Segment] = []
    in_motion = False
    start_idx = 0
    quiet_start_time: float | None = None

    for idx, row in work.iterrows():
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
            if current_time - quiet_start_time >= QUIET_DURATION:
                end_idx = int(idx)
                duration = float(work.loc[end_idx, "time_sec"] - work.loc[start_idx, "time_sec"])
                if duration >= MIN_DURATION:
                    segments.append(Segment(0, start_idx, end_idx))
                in_motion = False
                quiet_start_time = None

    if in_motion:
        end_idx = int(work.index[-1])
        duration = float(work.loc[end_idx, "time_sec"] - work.loc[start_idx, "time_sec"])
        if duration >= MIN_DURATION:
            segments.append(Segment(0, start_idx, end_idx))

    split_segments: list[Segment] = []
    for segment in segments:
        split_segments.extend(split_if_too_long(segment, work))

    numbered = [Segment(i + 1, segment.start_idx, segment.end_idx) for i, segment in enumerate(split_segments)]
    return numbered, baseline


def dominant_axis(window: pd.DataFrame) -> str:
    ranges = {
        "x": float(window["x"].max() - window["x"].min()),
        "y": float(window["y"].max() - window["y"].min()),
        "z": float(window["z"].max() - window["z"].min()),
    }
    return max(ranges, key=ranges.get)


def label_candidate(axis: str, peak_magnitude: float, baseline: float) -> str:
    if peak_magnitude - baseline >= 8.0:
        return "strong_motion"
    if axis == "x":
        return "horizontal_motion"
    if axis == "y":
        return "vertical_motion"
    return "unknown"


def build_segment_rows(
    df: pd.DataFrame,
    segments: list[Segment],
    baseline: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for segment in segments:
        window = df.loc[segment.start_idx : segment.end_idx]
        axis = dominant_axis(window)
        start_time = float(window["time_sec"].iloc[0])
        end_time = float(window["time_sec"].iloc[-1])
        peak_magnitude = float(window["magnitude"].max())
        duration = end_time - start_time
        rows.append(
            {
                "segment_id": segment.segment_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "peak_magnitude": peak_magnitude,
                "mean_magnitude": float(window["magnitude"].mean()),
                "dominant_axis": axis,
                "max_x": float(window["x"].max()),
                "max_y": float(window["y"].max()),
                "max_z": float(window["z"].max()),
                "mean_x": float(window["x"].mean()),
                "mean_y": float(window["y"].mean()),
                "mean_z": float(window["z"].mean()),
                "var_x": float(np.var(window["x"], ddof=0)),
                "var_y": float(np.var(window["y"], ddof=0)),
                "var_z": float(np.var(window["z"], ddof=0)),
                "label_candidate": label_candidate(axis, peak_magnitude, baseline),
            }
        )
    return rows


def summarize_threshold(
    df: pd.DataFrame,
    threshold: float,
) -> tuple[dict[str, float | int], list[Segment], float]:
    segments, baseline = detect_segments(df, threshold)
    rows = build_segment_rows(df, segments, baseline)
    durations = [float(row["duration"]) for row in rows]
    peaks = [float(row["peak_magnitude"]) for row in rows]
    summary = {
        "threshold": threshold,
        "segment_count": len(segments),
        "average_duration": float(np.mean(durations)) if durations else 0.0,
        "max_peak_magnitude": float(np.max(peaks)) if peaks else 0.0,
        "mean_peak_magnitude": float(np.mean(peaks)) if peaks else 0.0,
        "baseline_magnitude": baseline,
    }
    return summary, segments, baseline


def choose_best_threshold(comparison: list[dict[str, float | int]]) -> float:
    middle_threshold = float(np.median(THRESHOLDS))

    def score(row: dict[str, float | int]) -> tuple[float, float, float]:
        threshold = float(row["threshold"])
        count = int(row["segment_count"])
        if 1 <= count <= 10:
            count_penalty = 0.0
        elif count == 0:
            count_penalty = 100.0
        else:
            count_penalty = 50.0 + (count - 10)
        too_many_penalty = max(0, count - 10) * 2.0
        middle_penalty = abs(threshold - middle_threshold)
        return (count_penalty + too_many_penalty, middle_penalty, threshold)

    return float(min(comparison, key=score)["threshold"])


def save_comparison(rows: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "threshold",
        "segment_count",
        "average_duration",
        "max_peak_magnitude",
        "mean_peak_magnitude",
        "baseline_magnitude",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def plot_segments_overlay(
    df: pd.DataFrame,
    segments: list[Segment],
    baseline: float,
    threshold: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(13, 8))
    plt.plot(df["time_sec"], df["x"], label="x", linewidth=1.0, alpha=0.8)
    plt.plot(df["time_sec"], df["y"], label="y", linewidth=1.0, alpha=0.8)
    plt.plot(df["time_sec"], df["z"], label="z", linewidth=1.0, alpha=0.8)
    plt.plot(df["time_sec"], df["magnitude"], label="magnitude", linewidth=1.5, color="black")
    plt.axhline(baseline + threshold, color="red", linestyle="--", linewidth=1, label="baseline + threshold")
    plt.axhline(max(0.0, baseline - threshold), color="red", linestyle=":", linewidth=1, label="baseline - threshold")

    y_label = float(df["magnitude"].max()) if not df.empty else 0.0
    for segment in segments:
        start = float(df.loc[segment.start_idx, "time_sec"])
        end = float(df.loc[segment.end_idx, "time_sec"])
        plt.axvspan(start, end, color="orange", alpha=0.22)
        plt.text(start, y_label, f"#{segment.segment_id}", va="top", fontsize=9)

    plt.xlabel("time [sec]")
    plt.ylabel("acceleration")
    plt.title(f"Best detected segments (threshold={threshold})")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> int:
    root = backend_dir()
    raw_dir = root / "data" / "raw"
    output_dir = root / "data" / "processed" / "latest_analysis"

    try:
        input_path = find_latest_csv(raw_dir)
        df = load_log(input_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_raw_waveform(df, output_dir / "raw_waveform.png")
        plot_magnitude_waveform(df, output_dir / "magnitude_waveform.png")

        comparison: list[dict[str, float | int]] = []
        detected_by_threshold: dict[float, tuple[list[Segment], float]] = {}
        for threshold in THRESHOLDS:
            summary, segments, baseline = summarize_threshold(df, threshold)
            comparison.append(summary)
            detected_by_threshold[threshold] = (segments, baseline)

        save_comparison(comparison, output_dir / "comparison.csv")

        best_threshold = choose_best_threshold(comparison)
        best_segments, best_baseline = detected_by_threshold[best_threshold]
        best_rows = build_segment_rows(df, best_segments, best_baseline)
        save_segments(best_rows, output_dir / "best_segments.csv")
        plot_segments_overlay(
            df,
            best_segments,
            best_baseline,
            best_threshold,
            output_dir / "best_segments_overlay.png",
        )

        duration = float(df["time_sec"].iloc[-1] - df["time_sec"].iloc[0]) if len(df) > 1 else 0.0
        print("Latest log analysis complete")
        print(f"input CSV path: {input_path}")
        print(f"sample count: {len(df)}")
        print(f"duration: {duration:.3f} sec")
        print(f"selected best threshold: {best_threshold}")
        print(f"detected segment count: {len(best_segments)}")
        print(f"output directory: {output_dir}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
