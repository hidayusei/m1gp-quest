from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


SERIES_LENGTH = 32
ALLOWED_LABELS = {"circle", "star", "zigzag"}

BASIC_FEATURE_COLUMNS = [
    "duration",
    "x_range",
    "y_range",
    "z_range",
    "x_std",
    "y_std",
    "z_std",
    "magnitude_mean",
    "magnitude_max",
    "magnitude_std",
    "x_jerk_max",
    "y_jerk_max",
    "z_jerk_max",
    "x_y_correlation",
    "path_length_xy",
    "bounding_box_ratio",
    "direction_changes",
    "zero_crossings_x",
    "zero_crossings_y",
    "clockwise_score",
    "curvature_roughness",
    "number_of_turns",
    "start_end_distance",
    "closedness",
]

SERIES_FEATURE_COLUMNS = [
    *[f"x_series_{index}" for index in range(SERIES_LENGTH)],
    *[f"y_series_{index}" for index in range(SERIES_LENGTH)],
]

FEATURE_COLUMNS = [*BASIC_FEATURE_COLUMNS, *SERIES_FEATURE_COLUMNS]


def coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_time_seconds(timestamp: float, first_timestamp: float) -> float:
    value = timestamp - first_timestamp
    if abs(value) > 10_000:
        return value / 1000.0
    return value


def read_motion_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            x = coerce_float(row.get("x"))
            y = coerce_float(row.get("y"))
            z = coerce_float(row.get("z"))
            magnitude_value = row.get("magnitude")
            magnitude = (
                coerce_float(magnitude_value)
                if magnitude_value not in {None, ""}
                else math.sqrt(x * x + y * y + z * z)
            )
            rows.append(
                {
                    "timestamp": coerce_float(row.get("timestamp")),
                    "x": x,
                    "y": y,
                    "z": z,
                    "magnitude": magnitude,
                }
            )
    return rows


def value_range(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def max_abs_diff(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return max(abs(values[index] - values[index - 1]) for index in range(1, len(values)))


def safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return pstdev(values)


def safe_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or safe_std(xs) == 0 or safe_std(ys) == 0:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else 0.0


def normalize_axis(values: list[float]) -> list[float]:
    if not values:
        return []
    centered = [value - mean(values) for value in values]
    scale = max(max(abs(value) for value in centered), 1e-6)
    return [value / scale for value in centered]


def resample(values: list[float], length: int = SERIES_LENGTH) -> list[float]:
    if not values:
        return [0.0] * length
    if len(values) == 1:
        return [values[0]] * length
    result: list[float] = []
    last_index = len(values) - 1
    for output_index in range(length):
        position = output_index * last_index / (length - 1)
        left = int(math.floor(position))
        right = min(last_index, left + 1)
        ratio = position - left
        result.append(values[left] * (1.0 - ratio) + values[right] * ratio)
    return result


def zero_crossings(values: list[float]) -> int:
    if len(values) < 2:
        return 0
    crossings = 0
    previous = values[0]
    for value in values[1:]:
        if (previous <= 0 < value) or (previous >= 0 > value):
            crossings += 1
        previous = value
    return crossings


def xy_path_stats(xs: list[float], ys: list[float]) -> dict[str, float]:
    if len(xs) < 2:
        return {
            "path_length_xy": 0.0,
            "direction_changes": 0.0,
            "clockwise_score": 0.0,
            "curvature_roughness": 0.0,
            "number_of_turns": 0.0,
            "start_end_distance": 0.0,
            "closedness": 0.0,
        }

    path_length = 0.0
    angles: list[float] = []
    signed_area = 0.0
    for index in range(1, len(xs)):
        dx = xs[index] - xs[index - 1]
        dy = ys[index] - ys[index - 1]
        distance = math.hypot(dx, dy)
        path_length += distance
        if distance > 1e-6:
            angles.append(math.atan2(dy, dx))
        signed_area += xs[index - 1] * ys[index] - xs[index] * ys[index - 1]

    direction_changes = 0
    angle_deltas: list[float] = []
    for index in range(1, len(angles)):
        delta = math.atan2(math.sin(angles[index] - angles[index - 1]), math.cos(angles[index] - angles[index - 1]))
        angle_deltas.append(delta)
        if abs(delta) > math.radians(55):
            direction_changes += 1

    total_turn = sum(angle_deltas)
    start_end_distance = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    return {
        "path_length_xy": path_length,
        "direction_changes": float(direction_changes),
        "clockwise_score": 1.0 if signed_area < 0 else -1.0 if signed_area > 0 else 0.0,
        "curvature_roughness": safe_std(angle_deltas) if len(angle_deltas) >= 2 else 0.0,
        "number_of_turns": abs(total_turn) / (2.0 * math.pi),
        "start_end_distance": start_end_distance,
        "closedness": start_end_distance / max(path_length, 1e-6),
    }


def extract_features_from_samples(samples: list[dict[str, Any]]) -> dict[str, float]:
    cleaned = [
        {
            "timestamp": coerce_float(sample.get("timestamp")),
            "x": coerce_float(sample.get("x")),
            "y": coerce_float(sample.get("y")),
            "z": coerce_float(sample.get("z")),
            "magnitude": coerce_float(sample.get("magnitude")),
        }
        for sample in samples
    ]
    if not cleaned:
        return {name: 0.0 for name in FEATURE_COLUMNS}

    cleaned.sort(key=lambda sample: sample["timestamp"])
    xs_raw = [sample["x"] for sample in cleaned]
    ys_raw = [sample["y"] for sample in cleaned]
    zs = [sample["z"] for sample in cleaned]
    xs = normalize_axis(xs_raw)
    ys = normalize_axis(ys_raw)
    magnitudes = [
        sample["magnitude"] if sample["magnitude"] else math.sqrt(sample["x"] ** 2 + sample["y"] ** 2 + sample["z"] ** 2)
        for sample in cleaned
    ]
    first_timestamp = cleaned[0]["timestamp"]
    last_timestamp = cleaned[-1]["timestamp"]
    duration = normalize_time_seconds(last_timestamp, first_timestamp) if len(cleaned) >= 2 else 0.0
    x_range = value_range(xs_raw)
    y_range = value_range(ys_raw)
    bounding_box_ratio = x_range / max(y_range, 1e-6)
    path_stats = xy_path_stats(xs, ys)
    x_series = resample(xs)
    y_series = resample(ys)

    features: dict[str, float] = {
        "duration": max(0.0, duration),
        "x_range": x_range,
        "y_range": y_range,
        "z_range": value_range(zs),
        "x_std": safe_std(xs_raw),
        "y_std": safe_std(ys_raw),
        "z_std": safe_std(zs),
        "magnitude_mean": mean(magnitudes),
        "magnitude_max": max(magnitudes),
        "magnitude_std": safe_std(magnitudes),
        "x_jerk_max": max_abs_diff(xs_raw),
        "y_jerk_max": max_abs_diff(ys_raw),
        "z_jerk_max": max_abs_diff(zs),
        "x_y_correlation": safe_corr(xs_raw, ys_raw),
        "bounding_box_ratio": min(bounding_box_ratio, 999.0),
        "zero_crossings_x": float(zero_crossings(xs)),
        "zero_crossings_y": float(zero_crossings(ys)),
        **path_stats,
    }
    for index, value in enumerate(x_series):
        features[f"x_series_{index}"] = value
    for index, value in enumerate(y_series):
        features[f"y_series_{index}"] = value
    return features


def features_to_vector(features: dict[str, float]) -> list[float]:
    return [coerce_float(features.get(column, 0.0)) for column in FEATURE_COLUMNS]


def collect_training_features(training_dir: Path) -> tuple[list[list[float]], list[str], list[dict[str, Any]]]:
    vectors: list[list[float]] = []
    labels: list[str] = []
    rows: list[dict[str, Any]] = []
    if not training_dir.exists():
        return vectors, labels, rows
    for label_dir in sorted(path for path in training_dir.iterdir() if path.is_dir()):
        label = label_dir.name
        if label not in ALLOWED_LABELS:
            continue
        for csv_path in sorted(label_dir.glob("*.csv")):
            samples = read_motion_csv(csv_path)
            if not samples:
                continue
            features = extract_features_from_samples(samples)
            vector = features_to_vector(features)
            vectors.append(vector)
            labels.append(label)
            rows.append({"label": label, "path": str(csv_path), **features})
    return vectors, labels, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract shape motion features from training CSV files.")
    parser.add_argument("--training-dir", default=str(Path(__file__).resolve().parents[1] / "data" / "training"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "data" / "processed" / "training_features.csv"))
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not training_dir.exists():
        raise SystemExit(f"Training directory not found: {training_dir}")

    _, _, rows = collect_training_features(training_dir)
    fieldnames = ["label", "path", *FEATURE_COLUMNS]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved features: {output_path}")
    print(f"sample count: {len(rows)}")


if __name__ == "__main__":
    main()
