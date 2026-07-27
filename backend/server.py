from __future__ import annotations

import csv
import json
import math
import mimetypes
import pickle
import re
import shutil
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = BASE_DIR.parent / "frontend" / "dist"
RAW_DIR = BASE_DIR / "data" / "raw"
TRAINING_DIR = BASE_DIR / "data" / "training"
TRIGGER_TRAINING_DIR = BASE_DIR / "data" / "training_trigger"
TRIGGER_TRAINING_ARCHIVE_DIR = BASE_DIR / "data" / "training_trigger_archive"
MODELS_DIR = BASE_DIR / "models"
SHAPE_MODEL_PATH = MODELS_DIR / "shape_classifier.pkl"
TRIGGER_MODEL_PATH = MODELS_DIR / "trigger_detector.pkl"
CSV_COLUMNS = ["timestamp", "x", "y", "z", "magnitude"]
TRAINING_CSV_COLUMNS = ["timestamp", "x", "y", "z", "magnitude", "session_id"]
MODEL_SHAPE_LABELS = {"circle", "star", "triangle", "square", "zigzag", "none"}
ACTIVE_SHAPE_LABELS = {"circle", "star", "zigzag"}
SHAPE_LABELS = MODEL_SHAPE_LABELS
TRAINING_LABELS = ACTIVE_SHAPE_LABELS
TRIGGER_LABELS = {"trigger", "none"}
MAX_SESSION_SAMPLES = 12000
THRUST_WINDOW_MS = 900.0
THRUST_COOLDOWN_MS = 1800.0
START_STOP_COOLDOWN_MS = 1800.0
TRIGGER_ARM_DELAY_MS = 650.0
MAX_COMMAND_SECONDS = 12.0
TAP_MIN_RECORDING_MS = 900.0
TRIGGER_MODE = "strict candidate + model"
TRIGGER_MODEL_ENABLED = True
TRIGGER_AXIS = "z"
PRE_STABLE_MS = 450.0
POST_STABLE_MS = 260.0
TRIGGER_WINDOW_MS = 900.0
DYNAMIC_EMA_ALPHA = 0.08
DYNAMIC_MAG_STABLE_THRESHOLD = 0.9
STABLE_JERK_THRESHOLD = 0.8
DYNAMIC_PEAK_THRESHOLD = 0.55
PEAK_TO_PEAK_THRESHOLD = 1.05
TRIGGER_IMPULSE_THRESHOLD = 1.05
MIN_TRIGGER_PEAK_SEPARATION_MS = 90.0
MAX_DIRECTION_CHANGES = 4
TRIGGER_AXIS_DOMINANCE_RATIO = 0.9
TRIGGER_MODEL_CONFIDENCE_THRESHOLD = 0.6
AXIS_RANGE_THRESHOLDS = {"x": 1.4, "y": 2.8, "z": 0.9}
AXIS_JERK_THRESHOLDS = {"x": 0.45, "y": 0.9, "z": 0.35}
AXIS_SPEED_THRESHOLDS = {"x": 5.0, "y": 8.0, "z": 4.0}
LIVE_MIN_SCORE = 1.0
LIVE_MIN_SCORE_RATIO = 1.2
SHAPE_CONFIDENCE_THRESHOLD = 0.0
SQUARE_CONFIDENCE_THRESHOLD = 0.0
DTW_RESAMPLE_POINTS = 64
DTW_TRIM_RATIO = 0.06
DTW_K_NEIGHBORS = 5
DTW_WARP_WINDOW = 12
DTW_MIN_MARGIN = 0.08
DTW_DEFAULT_MAX_DISTANCE = 0.9
DTW_DISTANCE_SCALE = 1.35
DTW_LABEL_DIRECTORIES = {
    "circle": ("circle", "water_spiral"),
    "star": ("star", "fire_star"),
    "zigzag": ("zigzag", "thunder_z"),
    "none": ("none",),
}

SESSION_SAMPLES: dict[str, list[dict[str, float]]] = {}
ACTIVE_INPUTS: dict[str, dict[str, Any]] = {}
MOTION_INPUT_MODES: dict[str, str] = {}
ACTIVE_TRAINING: dict[str, dict[str, Any]] = {}
LAST_TRIGGER_CANDIDATES: dict[str, dict[str, Any]] = {}
LAST_SHAPE_CANDIDATES: dict[str, dict[str, Any]] = {}
SHAPE_MODEL_ARTIFACT: dict[str, Any] | None = None
TRIGGER_MODEL_ARTIFACT: dict[str, Any] | None = None
DTW_TEMPLATE_CACHE: list[dict[str, Any]] = []
DTW_TEMPLATE_SIGNATURE: tuple[tuple[str, int, int], ...] = ()
DTW_DISTANCE_THRESHOLDS: dict[str, float] = {}
SHAPE_MODEL_STATUS = "shape model not trained"
TRIGGER_MODEL_STATUS = "trigger fallback"
STATE_LOCK = Lock()

SHAPE_SKILLS = {
    "circle": {"displayName": "Water Spiral", "shape": "circle", "element": "water", "power": 50, "score": 50},
    "star": {"displayName": "Fire Star", "shape": "star", "element": "fire", "power": 50, "score": 50},
    "zigzag": {"displayName": "Thunder Z", "shape": "zigzag", "element": "thunder", "power": 50, "score": 50},
}
NO_SKILL = {"displayName": "No Skill", "shape": "none", "element": "water", "power": 0, "score": 0}


def safe_session_id(value: str | None) -> str:
    if not value:
        return "session"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return safe[:80] or "session"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def csv_path_for_session(session_id: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(RAW_DIR.glob(f"*_{session_id}.csv"))
    if candidates:
        return candidates[-1]
    return RAW_DIR / f"{now_stamp()}_{session_id}.csv"


def safe_training_label(value: str | None) -> str:
    label = str(value or "").strip()
    if label not in TRAINING_LABELS:
        raise ValueError("label must be one of circle, star, zigzag")
    return label


def safe_trigger_label(value: str | None) -> str:
    label = str(value or "").strip()
    if label not in TRIGGER_LABELS:
        raise ValueError("label must be trigger or none")
    return label


def safe_motion_input_mode(value: str | None) -> str:
    mode = str(value or "trigger").strip().lower()
    if mode not in {"trigger", "tap"}:
        raise ValueError("mode must be trigger or tap")
    return mode


def training_counts() -> dict[str, int]:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for label in sorted(TRAINING_LABELS):
        label_dir = TRAINING_DIR / label
        counts[label] = len(list(label_dir.glob("*.csv"))) if label_dir.exists() else 0
    return counts


def trigger_training_counts() -> dict[str, int]:
    TRIGGER_TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for label in sorted(TRIGGER_LABELS):
        label_dir = TRIGGER_TRAINING_DIR / label
        counts[label] = len(list(label_dir.glob("*.csv"))) if label_dir.exists() else 0
    legacy_p = TRAINING_DIR / "P"
    legacy_none = [TRAINING_DIR / "H", TRAINING_DIR / "V", TRAINING_DIR / "none"]
    counts["legacy_P_as_trigger"] = len(list(legacy_p.glob("*.csv"))) if legacy_p.exists() else 0
    counts["legacy_none"] = sum(len(list(path.glob("*.csv"))) for path in legacy_none if path.exists())
    return counts


def next_training_csv_path(label: str) -> Path:
    label_dir = TRAINING_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(label_dir.glob(f"{label}_*.csv"))
    return label_dir / f"{label}_{len(existing) + 1:03d}.csv"


def next_trigger_training_csv_path(label: str) -> Path:
    label_dir = TRIGGER_TRAINING_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(label_dir.glob(f"{label}_*.csv"))
    return label_dir / f"{label}_{len(existing) + 1:03d}.csv"


def archive_trigger_training_data() -> dict[str, Any]:
    archive_dir = TRIGGER_TRAINING_ARCHIVE_DIR / now_stamp()
    suffix = 1
    while archive_dir.exists():
        archive_dir = TRIGGER_TRAINING_ARCHIVE_DIR / f"{now_stamp()}_{suffix}"
        suffix += 1

    moved_counts: dict[str, int] = {}
    moved_files = 0
    for label in sorted(TRIGGER_LABELS):
        source_dir = TRIGGER_TRAINING_DIR / label
        target_dir = archive_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        if source_dir.exists():
            for csv_path in sorted(source_dir.glob("*.csv")):
                shutil.move(str(csv_path), str(target_dir / csv_path.name))
                count += 1
                moved_files += 1
        moved_counts[label] = count
        source_dir.mkdir(parents=True, exist_ok=True)

    model_archived = False
    if TRIGGER_MODEL_PATH.exists():
        model_dir = archive_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(TRIGGER_MODEL_PATH), str(model_dir / TRIGGER_MODEL_PATH.name))
        model_archived = True

    return {
        "archiveDir": str(archive_dir),
        "movedCounts": moved_counts,
        "movedFiles": moved_files,
        "modelArchived": model_archived,
    }


def load_shape_model() -> None:
    global SHAPE_MODEL_ARTIFACT, SHAPE_MODEL_STATUS
    if not SHAPE_MODEL_PATH.exists():
        SHAPE_MODEL_ARTIFACT = None
        SHAPE_MODEL_STATUS = "shape model not trained"
        return
    try:
        with SHAPE_MODEL_PATH.open("rb") as model_file:
            artifact = pickle.load(model_file)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ValueError("invalid model artifact")
        labels = set(str(label) for label in artifact.get("labels", []))
        if not labels or any(label not in SHAPE_LABELS for label in labels):
            raise ValueError("model is not a shape motion classifier")
        SHAPE_MODEL_ARTIFACT = artifact
        SHAPE_MODEL_STATUS = "enabled"
    except Exception as exc:
        SHAPE_MODEL_ARTIFACT = None
        if "shape motion classifier" in str(exc):
            SHAPE_MODEL_STATUS = "shape model not trained"
        else:
            SHAPE_MODEL_STATUS = f"load failed: {exc}"


def load_trigger_model() -> None:
    global TRIGGER_MODEL_ARTIFACT, TRIGGER_MODEL_STATUS
    if not TRIGGER_MODEL_PATH.exists():
        TRIGGER_MODEL_ARTIFACT = None
        TRIGGER_MODEL_STATUS = "trigger fallback"
        return
    try:
        with TRIGGER_MODEL_PATH.open("rb") as model_file:
            artifact = pickle.load(model_file)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ValueError("invalid trigger artifact")
        labels = set(str(label) for label in artifact.get("labels", []))
        if not labels or any(label not in TRIGGER_LABELS for label in labels):
            raise ValueError("model is not a trigger detector")
        TRIGGER_MODEL_ARTIFACT = artifact
        TRIGGER_MODEL_STATUS = "enabled"
    except Exception as exc:
        TRIGGER_MODEL_ARTIFACT = None
        TRIGGER_MODEL_STATUS = f"load failed: {exc}"


def load_motion_model() -> None:
    load_shape_model()
    load_trigger_model()


def model_status_payload() -> dict[str, Any]:
    return {
        "modelExists": SHAPE_MODEL_ARTIFACT is not None,
        "modelFileExists": SHAPE_MODEL_PATH.exists(),
        "modelPath": "backend/models/shape_classifier.pkl",
        "modelStatus": SHAPE_MODEL_STATUS,
        "modelLabels": (SHAPE_MODEL_ARTIFACT or {}).get("labels", []),
        "shapeModelExists": SHAPE_MODEL_ARTIFACT is not None,
        "shapeModelPath": "backend/models/shape_classifier.pkl",
        "shapeModelStatus": SHAPE_MODEL_STATUS,
        "shapeModelLabels": (SHAPE_MODEL_ARTIFACT or {}).get("labels", []),
        "triggerModelExists": TRIGGER_MODEL_ARTIFACT is not None,
        "triggerModelPath": "backend/models/trigger_detector.pkl",
        "triggerModelStatus": TRIGGER_MODEL_STATUS,
        "triggerModelLabels": (TRIGGER_MODEL_ARTIFACT or {}).get("labels", []),
    }


def coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def trim_and_resample_dtw(samples: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    ordered = sorted(samples, key=lambda sample: coerce_float(sample.get("timestamp")))
    if len(ordered) < 2:
        return []

    trim_count = int(len(ordered) * DTW_TRIM_RATIO)
    if trim_count > 0 and len(ordered) - trim_count * 2 >= 8:
        ordered = ordered[trim_count:-trim_count]

    axes = [
        [coerce_float(sample.get(axis)) for sample in ordered]
        for axis in ("x", "y", "z")
    ]

    def resample(values: list[float]) -> list[float]:
        if len(values) == 1:
            return values * DTW_RESAMPLE_POINTS
        result: list[float] = []
        last_index = len(values) - 1
        for output_index in range(DTW_RESAMPLE_POINTS):
            position = output_index * last_index / (DTW_RESAMPLE_POINTS - 1)
            left = int(math.floor(position))
            right = min(last_index, left + 1)
            ratio = position - left
            result.append(values[left] * (1.0 - ratio) + values[right] * ratio)
        return result

    normalized_axes: list[list[float]] = []
    for values in axes:
        sampled = resample(values)
        axis_mean = sum(sampled) / len(sampled)
        variance = sum((value - axis_mean) ** 2 for value in sampled) / len(sampled)
        axis_std = math.sqrt(variance)
        if axis_std < 1e-6:
            normalized_axes.append([0.0] * len(sampled))
        else:
            normalized_axes.append([(value - axis_mean) / axis_std for value in sampled])

    return list(zip(*normalized_axes))


def multivariate_dtw_distance(
    first: list[tuple[float, float, float]],
    second: list[tuple[float, float, float]],
) -> float:
    if not first or not second:
        return math.inf
    first_length = len(first)
    second_length = len(second)
    window = max(DTW_WARP_WINDOW, abs(first_length - second_length))
    previous = [math.inf] * (second_length + 1)
    previous[0] = 0.0

    for first_index in range(1, first_length + 1):
        current = [math.inf] * (second_length + 1)
        start = max(1, first_index - window)
        end = min(second_length, first_index + window)
        for second_index in range(start, end + 1):
            left = first[first_index - 1]
            right = second[second_index - 1]
            point_distance = math.sqrt(
                ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2 + (left[2] - right[2]) ** 2)
                / 3.0
            )
            current[second_index] = point_distance + min(
                current[second_index - 1],
                previous[second_index],
                previous[second_index - 1],
            )
        previous = current

    return previous[second_length] / max(first_length + second_length, 1)


def read_dtw_csv(path: Path) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            samples.append(
                {
                    "timestamp": coerce_float(row.get("timestamp")),
                    "x": coerce_float(row.get("x")),
                    "y": coerce_float(row.get("y")),
                    "z": coerce_float(row.get("z")),
                }
            )
    return samples


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return DTW_DEFAULT_MAX_DISTANCE
    ordered = sorted(values)
    position = max(0.0, min(1.0, ratio)) * (len(ordered) - 1)
    left = int(math.floor(position))
    right = min(len(ordered) - 1, left + 1)
    fraction = position - left
    return ordered[left] * (1.0 - fraction) + ordered[right] * fraction


def refresh_dtw_templates() -> list[dict[str, Any]]:
    global DTW_TEMPLATE_CACHE, DTW_TEMPLATE_SIGNATURE, DTW_DISTANCE_THRESHOLDS
    labeled_paths: list[tuple[str, Path]] = []
    for label, directory_names in DTW_LABEL_DIRECTORIES.items():
        for directory_name in directory_names:
            label_dir = TRAINING_DIR / directory_name
            if label_dir.exists():
                labeled_paths.extend((label, path) for path in sorted(label_dir.glob("*.csv")))

    signature = tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for _, path in labeled_paths
    )
    if signature == DTW_TEMPLATE_SIGNATURE:
        return DTW_TEMPLATE_CACHE

    templates: list[dict[str, Any]] = []
    for label, path in labeled_paths:
        try:
            series = trim_and_resample_dtw(read_dtw_csv(path))
            if series:
                templates.append({"label": label, "path": str(path), "series": series})
        except (OSError, csv.Error, ValueError):
            continue

    thresholds: dict[str, float] = {}
    for label in DTW_LABEL_DIRECTORIES:
        label_templates = [template for template in templates if template["label"] == label]
        nearest_distances: list[float] = []
        for index, template in enumerate(label_templates):
            candidates = [
                multivariate_dtw_distance(template["series"], other["series"])
                for other_index, other in enumerate(label_templates)
                if other_index != index
            ]
            if candidates:
                nearest_distances.append(min(candidates))
        calibrated = percentile(nearest_distances, 0.9) * DTW_DISTANCE_SCALE
        thresholds[label] = max(0.12, calibrated) if nearest_distances else DTW_DEFAULT_MAX_DISTANCE

    DTW_TEMPLATE_CACHE = templates
    DTW_TEMPLATE_SIGNATURE = signature
    DTW_DISTANCE_THRESHOLDS = thresholds
    return DTW_TEMPLATE_CACHE


def predict_shape_dtw(samples: list[dict[str, Any]]) -> dict[str, Any]:
    templates = refresh_dtw_templates()
    series = trim_and_resample_dtw(samples)
    active_templates = [template for template in templates if template["label"] in ACTIVE_SHAPE_LABELS]
    if not series or not active_templates:
        return {"available": False, "reason": "DTW templates unavailable"}

    distances = [
        {
            "label": str(template["label"]),
            "distance": multivariate_dtw_distance(series, template["series"]),
        }
        for template in active_templates
    ]
    distances.sort(key=lambda item: float(item["distance"]))

    distances_by_label: dict[str, list[float]] = {}
    for item in distances:
        label = str(item["label"])
        distances_by_label.setdefault(label, []).append(float(item["distance"]))

    # Compare the mean of each label's nearest templates. This is less sensitive
    # to one accidental template match and does not favor labels with more files.
    label_scores: dict[str, float] = {}
    per_label_neighbors = max(1, min(3, DTW_K_NEIGHBORS))
    for label, label_distances in distances_by_label.items():
        nearest = sorted(label_distances)[:per_label_neighbors]
        label_scores[label] = sum(nearest) / len(nearest)

    label_ranking = sorted(label_scores.items(), key=lambda item: item[1])
    predicted_label, top1_distance = label_ranking[0]
    top2_label, top2_distance = label_ranking[1] if len(label_ranking) > 1 else (predicted_label, top1_distance)

    margin = max(0.0, (top2_distance - top1_distance) / max(top2_distance, 1e-9))
    max_distance = DTW_DISTANCE_THRESHOLDS.get(predicted_label, DTW_DEFAULT_MAX_DISTANCE)
    distance_score = max(0.0, 1.0 - top1_distance / max(max_distance, 1e-9))
    confidence = max(0.0, min(1.0, margin * 0.7 + distance_score * 0.3))
    warning = ""
    if top1_distance > max_distance:
        warning = "low confidence: DTW distance above threshold"
    elif margin < DTW_MIN_MARGIN:
        warning = "low confidence: DTW top labels too close"

    return {
        "available": True,
        "label": predicted_label,
        "confidence": confidence,
        "top1Label": predicted_label,
        "top1Distance": top1_distance,
        "top2Label": top2_label,
        "top2Distance": top2_distance,
        "distanceThreshold": max_distance,
        "margin": margin,
        "rejectedReason": warning,
        "templateCount": len(active_templates),
        "usedModel": "dtw",
    }


def now_ms() -> float:
    return time.time() * 1000.0


def sample_time_ms(timestamp: float, fallback_received_at: float) -> float:
    if timestamp <= 0:
        return fallback_received_at
    if timestamp < 10_000_000_000:
        return timestamp * 1000.0
    return timestamp


def choose_shape_skill(label: str | None) -> dict[str, Any]:
    if label in SHAPE_SKILLS:
        return dict(SHAPE_SKILLS[str(label)])
    return dict(NO_SKILL)


def active_samples_for_state(session_id: str, state: dict[str, Any]) -> list[dict[str, float]]:
    started_at = float(state["startedAt"])
    ends_at = float(state["endsAt"])
    samples = SESSION_SAMPLES.get(session_id, [])
    return [
        sample
        for sample in samples
        if started_at <= sample["received_at"] <= ends_at
    ]


def command_from_window(window: list[dict[str, float]]) -> tuple[str, dict[str, float]]:
    ordered = sorted(window, key=lambda sample: sample.get("time_ms", sample["received_at"]))
    ranges = {
        "x": value_range(window, "x"),
        "y": value_range(window, "y"),
        "z": value_range(window, "z"),
    }
    jerks = {
        "x": max_abs_diff(ordered, "x"),
        "y": max_abs_diff(ordered, "y"),
        "z": max_abs_diff(ordered, "z"),
    }
    magnitude_values = [sample["magnitude"] for sample in window]
    magnitude_range = max(magnitude_values) - min(magnitude_values)
    duration_ms = max(
        1.0,
        ordered[-1].get("time_ms", ordered[-1]["received_at"])
        - ordered[0].get("time_ms", ordered[0]["received_at"]),
    )
    speeds = {
        axis: ranges[axis] / (duration_ms / 1000.0)
        for axis in ("x", "y", "z")
    }
    scores = {
        axis: ranges[axis] / AXIS_RANGE_THRESHOLDS[axis]
        for axis in ("x", "y", "z")
    }
    ordered_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    dominant_axis, best_score = ordered_scores[0]
    second_score = ordered_scores[1][1]
    dominant_range = ranges[dominant_axis]
    dominant_jerk = jerks[dominant_axis]
    dominant_speed = speeds[dominant_axis]
    command_by_axis = {"x": "H", "y": "V", "z": "P"}
    rejected_reason = ""

    if best_score < LIVE_MIN_SCORE:
        command = "U"
        rejected_reason = "score below threshold"
    elif second_score > 0 and best_score < second_score * LIVE_MIN_SCORE_RATIO:
        command = "U"
        rejected_reason = "axis not dominant enough"
    elif dominant_jerk < AXIS_JERK_THRESHOLDS[dominant_axis]:
        command = "U"
        rejected_reason = "jerk too small"
    elif dominant_speed < AXIS_SPEED_THRESHOLDS[dominant_axis]:
        command = "U"
        rejected_reason = "movement too slow"
    else:
        command = command_by_axis[dominant_axis]

    model_label, model_confidence, model_status = predict_shape_label(window)
    if model_label in {"H", "V", "P"}:
        command = model_label
        rejected_reason = ""
    elif model_label == "none":
        command = "U"
        rejected_reason = "model predicted none"

    return command, {
        "rangeX": ranges["x"],
        "rangeY": ranges["y"],
        "rangeZ": ranges["z"],
        "jerkX": jerks["x"],
        "jerkY": jerks["y"],
        "jerkZ": jerks["z"],
        "scoreX": scores["x"],
        "scoreY": scores["y"],
        "scoreZ": scores["z"],
        "magnitudeRange": magnitude_range,
        "bestAxis": dominant_axis,
        "bestScore": best_score,
        "secondScore": second_score,
        "axisSpeed": dominant_speed,
        "selectedCommand": command,
        "rejectedReason": rejected_reason,
        "modelStatus": model_status,
        "modelLabel": model_label or "",
        "modelConfidence": model_confidence if model_confidence is not None else 0.0,
    }


def make_waiting_state(session_id: str) -> dict[str, Any]:
    control_mode = MOTION_INPUT_MODES.get(session_id, "trigger")
    return {
        "sessionId": session_id,
        "controlMode": control_mode,
        "inputState": "waiting_for_start",
        "isRecording": False,
        "armedAt": now_ms(),
        "startedAt": 0.0,
        "endsAt": 0.0,
        "serverSamples": 0,
        "recognizedShape": "",
        "confidence": 0.0,
        "selectedSkill": choose_shape_skill(None),
        "currentBestSkill": choose_shape_skill(None),
        "latestMetrics": {"message": "waiting for wand tap" if control_mode == "tap" else "waiting for cast trigger"},
        "lastTriggerCandidate": [],
        "lastTriggerCandidateAt": 0.0,
        "lastShapeSamples": [],
        "lastShapeSampleCount": 0,
        "lastShapeCapturedAt": 0.0,
        "triggerCooldownUntil": 0.0,
        "ignoreStopUntil": 0.0,
    }


def predict_trigger_label(window: list[dict[str, float]]) -> tuple[str | None, float | None, str]:
    if not TRIGGER_MODEL_ARTIFACT:
        return None, None, TRIGGER_MODEL_STATUS
    try:
        from scripts.extract_motion_features import extract_features_from_samples, features_to_vector

        model = TRIGGER_MODEL_ARTIFACT["model"]
        vector = [features_to_vector(extract_features_from_samples(window))]
        label = str(model.predict(vector)[0])
        confidence: float | None = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(vector)[0]
            confidence = float(max(probabilities))
        return label, confidence, "enabled"
    except Exception as exc:
        return None, None, f"trigger model error: {exc}"


def trigger_metrics(samples: list[dict[str, float]], current_time: float) -> tuple[bool, dict[str, float | str]]:
    history_start = current_time - PRE_STABLE_MS - TRIGGER_WINDOW_MS - POST_STABLE_MS - 400.0
    history = [
        sample
        for sample in samples
        if history_start <= sample.get("time_ms", sample["received_at"]) <= current_time
    ]
    if len(history) < 12:
        return False, {
            "autoTriggerMode": TRIGGER_MODE,
            "triggerModelStatus": TRIGGER_MODEL_STATUS if TRIGGER_MODEL_ARTIFACT is not None else "missing, strict fallback",
            "triggerState": "waiting_stable",
            "triggerGatePassed": False,
            "rejectedReason": "not enough samples for strict trigger",
        }

    ordered = sorted(history, key=lambda sample: sample.get("time_ms", sample["received_at"]))
    ema = {axis: ordered[0][axis] for axis in ("x", "y", "z")}
    dynamic: list[dict[str, float]] = []
    for sample in ordered:
        enriched = dict(sample)
        for axis in ("x", "y", "z"):
            value = sample[axis]
            enriched[f"raw_{axis}"] = value
            enriched[f"dynamic_{axis}"] = value - ema[axis]
            ema[axis] = ema[axis] + DYNAMIC_EMA_ALPHA * (value - ema[axis])
        enriched["dynamic_magnitude"] = math.sqrt(
            enriched["dynamic_x"] ** 2 + enriched["dynamic_y"] ** 2 + enriched["dynamic_z"] ** 2
        )
        dynamic.append(enriched)

    gesture_end = current_time - POST_STABLE_MS
    gesture_start = gesture_end - TRIGGER_WINDOW_MS
    pre_start = gesture_start - PRE_STABLE_MS
    pre_window = [sample for sample in dynamic if pre_start <= sample.get("time_ms", sample["received_at"]) < gesture_start]
    gesture_window = [sample for sample in dynamic if gesture_start <= sample.get("time_ms", sample["received_at"]) <= gesture_end]
    post_window = [sample for sample in dynamic if gesture_end < sample.get("time_ms", sample["received_at"]) <= current_time]

    def stable_metrics(window: list[dict[str, float]]) -> tuple[bool, float, float]:
        if len(window) < 4:
            return False, 999.0, 999.0
        dynamic_mag_mean = sum(sample["dynamic_magnitude"] for sample in window) / len(window)
        jerk_max = max(
            max_abs_diff(window, "dynamic_x"),
            max_abs_diff(window, "dynamic_y"),
            max_abs_diff(window, "dynamic_z"),
        )
        return (
            dynamic_mag_mean < DYNAMIC_MAG_STABLE_THRESHOLD and jerk_max < STABLE_JERK_THRESHOLD,
            dynamic_mag_mean,
            jerk_max,
        )

    pre_stable, pre_mag_mean, pre_jerk_max = stable_metrics(pre_window)
    post_stable, post_mag_mean, post_jerk_max = stable_metrics(post_window)

    axis_stats: dict[str, dict[str, float]] = {}
    for axis in ("x", "y", "z"):
        values = [sample[f"dynamic_{axis}"] for sample in gesture_window]
        times = [sample.get("time_ms", sample["received_at"]) for sample in gesture_window]
        if not values:
            axis_stats[axis] = {
                "positivePeak": 0.0,
                "negativePeak": 0.0,
                "peakToPeak": 0.0,
                "directionChange": 0.0,
                "impulse": 0.0,
                "jerk": 0.0,
                "peakSeparationMs": 0.0,
            }
            continue
        diffs = [values[index] - values[index - 1] for index in range(1, len(values))]
        peak_to_peak = max(values) - min(values)
        direction_epsilon = max(0.22, peak_to_peak * 0.08)
        signs = [1 if value > direction_epsilon else -1 if value < -direction_epsilon else 0 for value in diffs]
        compact_signs = [sign for sign in signs if sign != 0]
        direction_change = sum(
            1
            for index in range(1, len(compact_signs))
            if compact_signs[index - 1] != compact_signs[index]
        )
        positive_index = max(range(len(values)), key=lambda index: values[index])
        negative_index = min(range(len(values)), key=lambda index: values[index])
        axis_stats[axis] = {
            "positivePeak": values[positive_index],
            "negativePeak": values[negative_index],
            "peakToPeak": peak_to_peak,
            "directionChange": float(direction_change),
            "impulse": sum(abs(value) for value in diffs),
            "jerk": max((abs(value) for value in diffs), default=0.0),
            "peakSeparationMs": abs(times[positive_index] - times[negative_index]),
        }

    trigger_scores = {
        axis: (
            axis_stats[axis]["peakToPeak"] / PEAK_TO_PEAK_THRESHOLD
            + axis_stats[axis]["jerk"] / DYNAMIC_PEAK_THRESHOLD
            + axis_stats[axis]["impulse"] / TRIGGER_IMPULSE_THRESHOLD
        )
        for axis in ("z", "y")
    }
    selected_axis = max(trigger_scores, key=trigger_scores.get)
    stats = axis_stats[selected_axis]
    other_peak_to_peak = max(axis_stats[axis]["peakToPeak"] for axis in ("x", "y", "z") if axis != selected_axis)
    axis_dominant = stats["peakToPeak"] >= max(other_peak_to_peak, 0.001) * TRIGGER_AXIS_DOMINANCE_RATIO
    has_positive_peak = stats["positivePeak"] >= DYNAMIC_PEAK_THRESHOLD
    has_negative_peak = abs(stats["negativePeak"]) >= DYNAMIC_PEAK_THRESHOLD
    has_return_peak = (
        has_positive_peak
        and has_negative_peak
        and stats["peakSeparationMs"] >= MIN_TRIGGER_PEAK_SEPARATION_MS
    )

    trigger_state = "waiting_stable"
    reason = ""
    strict_candidate = False
    if not has_positive_peak and not has_negative_peak:
        trigger_state = "armed" if pre_stable else "waiting_stable"
        reason = "no trigger peak"
    elif not has_return_peak:
        trigger_state = "candidate_peak_detected"
        reason = "return peak missing"
    elif stats["peakToPeak"] < PEAK_TO_PEAK_THRESHOLD:
        trigger_state = "return_peak_detected"
        reason = "peak-to-peak below threshold"
    elif stats["impulse"] < TRIGGER_IMPULSE_THRESHOLD:
        trigger_state = "return_peak_detected"
        reason = "trigger impulse below threshold"
    elif not axis_dominant:
        trigger_state = "return_peak_detected"
        reason = "trigger axis not dominant"
    elif not post_stable:
        trigger_state = "waiting_settle"
        reason = "post stable required"
    else:
        trigger_state = "trigger_candidate"
        strict_candidate = True

    model_label: str | None = None
    model_confidence: float | None = None
    model_status = "not evaluated"
    detected = False
    final_decision = "rejected"
    if strict_candidate:
        if TRIGGER_MODEL_ENABLED and TRIGGER_MODEL_ARTIFACT is not None:
            model_label, model_confidence, model_status = predict_trigger_label(gesture_window)
            confidence_value = float(model_confidence or 0.0)
            if model_label == "trigger" and confidence_value >= TRIGGER_MODEL_CONFIDENCE_THRESHOLD:
                detected = True
                final_decision = "trigger"
                reason = ""
                trigger_state = "trigger_confirmed"
            elif model_label == "trigger":
                reason = "trigger model confidence below threshold"
                trigger_state = "candidate_rejected_by_model"
            else:
                reason = "trigger model predicted none"
                trigger_state = "candidate_rejected_by_model"
        else:
            fallback_ok = pre_stable and stats["directionChange"] <= MAX_DIRECTION_CHANGES
            detected = fallback_ok
            final_decision = "trigger_fallback" if fallback_ok else "rejected"
            reason = "" if fallback_ok else "strict fallback requires stable and simple return"
            model_status = "missing, strict fallback"
            trigger_state = "trigger_confirmed_fallback" if fallback_ok else "candidate_rejected_by_fallback"
    confidence_value = float(model_confidence or 0.0)

    latest = dynamic[-1]
    metrics: dict[str, Any] = {
        "autoTriggerMode": TRIGGER_MODE,
        "triggerModelStatus": model_status,
        "triggerAxis": selected_axis,
        "rawX": latest["raw_x"],
        "rawY": latest["raw_y"],
        "rawZ": latest["raw_z"],
        "dynamicX": latest["dynamic_x"],
        "dynamicY": latest["dynamic_y"],
        "dynamicZ": latest["dynamic_z"],
        "dynamicMagnitude": latest["dynamic_magnitude"],
        "rangeX": axis_stats["x"]["peakToPeak"],
        "rangeY": axis_stats["y"]["peakToPeak"],
        "rangeZ": axis_stats["z"]["peakToPeak"],
        "jerkX": axis_stats["x"]["jerk"],
        "jerkY": axis_stats["y"]["jerk"],
        "jerkZ": axis_stats["z"]["jerk"],
        "positivePeak": stats["positivePeak"],
        "negativePeak": stats["negativePeak"],
        "triggerPeakToPeak": stats["peakToPeak"],
        "triggerDirectionChange": stats["directionChange"],
        "triggerImpulse": stats["impulse"],
        "preStableMs": PRE_STABLE_MS if pre_stable else 0.0,
        "postStableMs": POST_STABLE_MS if post_stable else 0.0,
        "preStableMagMean": pre_mag_mean,
        "preStableJerkMax": pre_jerk_max,
        "postStableMagMean": post_mag_mean,
        "postStableJerkMax": post_jerk_max,
        "yDirectionChange": axis_stats["y"]["directionChange"],
        "zDirectionChange": axis_stats["z"]["directionChange"],
        "yImpulse": axis_stats["y"]["impulse"],
        "zImpulse": axis_stats["z"]["impulse"],
        "scoreY": trigger_scores["y"],
        "scoreZ": trigger_scores["z"],
        "triggerState": trigger_state,
        "strictGatePassed": bool(strict_candidate),
        "triggerCandidateExists": bool(strict_candidate),
        "triggerCandidateDuration": TRIGGER_WINDOW_MS if strict_candidate else 0.0,
        "triggerGatePassed": bool(detected),
        "finalTriggerDecision": final_decision,
        "selectedCommand": "trigger" if detected else "",
        "rejectedReason": reason,
        "triggerModelLabel": model_label or "",
        "triggerModelConfidence": confidence_value,
    }
    if strict_candidate:
        metrics["__candidateSamples"] = gesture_window
    return detected, metrics


def classify_shape_motion(samples: list[dict[str, float]]) -> dict[str, Any]:
    if len(samples) < 6:
        return {
            "recognizedShape": "none",
            "confidence": 0.0,
            "selectedSkill": choose_shape_skill(None),
            "message": "not enough samples",
            "modelStatus": "dtw rejected",
            "metrics": {
                "dtwTop1Label": "none",
                "dtwTop1Distance": 0.0,
                "dtwTop2Label": "none",
                "dtwTop2Distance": 0.0,
                "dtwConfidence": 0.0,
                "usedModel": "dtw",
                "dtwRejectedReason": "not enough samples",
            },
        }

    try:
        dtw_result = predict_shape_dtw(samples)
    except Exception as exc:
        dtw_result = {"available": False, "reason": f"DTW error: {exc}"}

    if dtw_result.get("available"):
        label = str(dtw_result.get("label") or "none")
        if label not in ACTIVE_SHAPE_LABELS and label != "none":
            label = "none"
        confidence_value = float(dtw_result.get("confidence") or 0.0)
        rejected_reason = str(dtw_result.get("rejectedReason") or "")
        return {
            "recognizedShape": label,
            "confidence": confidence_value,
            "selectedSkill": choose_shape_skill(label if label in ACTIVE_SHAPE_LABELS else None),
            "message": rejected_reason or "DTW template match",
            "modelStatus": "dtw",
            "metrics": {
                "dtwTop1Label": str(dtw_result.get("top1Label") or "none"),
                "dtwTop1Distance": float(dtw_result.get("top1Distance") or 0.0),
                "dtwTop2Label": str(dtw_result.get("top2Label") or "none"),
                "dtwTop2Distance": float(dtw_result.get("top2Distance") or 0.0),
                "dtwConfidence": confidence_value,
                "dtwDistanceThreshold": float(dtw_result.get("distanceThreshold") or 0.0),
                "dtwMargin": float(dtw_result.get("margin") or 0.0),
                "dtwTemplateCount": int(dtw_result.get("templateCount") or 0),
                "dtwRejectedReason": rejected_reason,
                "usedModel": "dtw",
            },
        }

    label, confidence, model_status = predict_shape_label(samples)
    if label not in ACTIVE_SHAPE_LABELS:
        label = "none"
    confidence_value = float(confidence or 0.0)
    return {
        "recognizedShape": label,
        "confidence": confidence_value,
        "selectedSkill": choose_shape_skill(label if label in ACTIVE_SHAPE_LABELS else None),
        "message": "RandomForest fallback",
        "modelStatus": f"random forest fallback: {model_status}",
        "metrics": {
            "dtwTop1Label": "-",
            "dtwTop1Distance": 0.0,
            "dtwTop2Label": "-",
            "dtwTop2Distance": 0.0,
            "dtwConfidence": 0.0,
            "dtwRejectedReason": str(dtw_result.get("reason") or "DTW unavailable"),
            "usedModel": "random_forest",
        },
    }


def training_samples_for_window(session_id: str, started_at: float, ended_at: float) -> list[dict[str, float]]:
    return [
        sample
        for sample in SESSION_SAMPLES.get(session_id, [])
        if started_at <= sample.get("time_ms", sample["received_at"]) <= ended_at
    ]


def save_training_samples(session_id: str, label: str, samples: list[dict[str, float]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("no samples found between cast start and cast confirm")
    path = next_training_csv_path(label)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRAINING_CSV_COLUMNS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "timestamp": sample["timestamp"],
                    "x": sample["x"],
                    "y": sample["y"],
                    "z": sample["z"],
                    "magnitude": sample["magnitude"],
                    "session_id": session_id,
                }
            )
    return {"savedPath": str(path), "sampleCount": len(samples), "counts": training_counts()}


def save_trigger_training_samples(session_id: str, label: str, samples: list[dict[str, float]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("no trigger training samples found")
    path = next_trigger_training_csv_path(label)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRAINING_CSV_COLUMNS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "timestamp": sample["timestamp"],
                    "x": sample["x"],
                    "y": sample["y"],
                    "z": sample["z"],
                    "magnitude": sample["magnitude"],
                    "session_id": session_id,
                }
            )
    return {"savedPath": str(path), "sampleCount": len(samples), "counts": trigger_training_counts()}


def finish_recording_state_locked(session_id: str, state: dict[str, Any], ended_at: float) -> None:
    started_at = float(state.get("startedAt") or 0.0)
    ended_at = max(started_at, ended_at)
    samples = training_samples_for_window(session_id, started_at, ended_at)
    result = classify_shape_motion(samples)
    latest_metrics = dict(state.get("latestMetrics") or {})
    latest_metrics.update(result.get("metrics") or {})
    state.update(
        {
            "inputState": "finished",
            "isRecording": False,
            "endsAt": ended_at,
            "serverSamples": len(samples),
            "recognizedShape": result["recognizedShape"],
            "confidence": result["confidence"],
            "selectedSkill": result["selectedSkill"],
            "currentBestSkill": result["selectedSkill"],
            "message": result["message"],
            "modelStatus": result["modelStatus"],
            "latestMetrics": latest_metrics,
            "lastShapeSamples": samples,
            "lastShapeSampleCount": len(samples),
            "lastShapeCapturedAt": ended_at,
        }
    )
    if samples:
        LAST_SHAPE_CANDIDATES[session_id] = {
            "samples": list(samples),
            "capturedAt": ended_at,
            "recognizedShape": result["recognizedShape"],
            "confidence": result["confidence"],
        }


def finish_training_state_locked(session_id: str, state: dict[str, Any], ended_at: float) -> None:
    label = safe_training_label(state.get("label"))
    started_at = float(state.get("startedAt") or 0.0)
    samples = training_samples_for_window(session_id, started_at, max(started_at, ended_at))
    saved = save_training_samples(session_id, label, samples)
    state.update(
        {
            "inputState": "finished",
            "isRecording": False,
            "endsAt": ended_at,
            "serverSamples": len(samples),
            "savedPath": saved["savedPath"],
            "sampleCount": saved["sampleCount"],
            "counts": saved["counts"],
            "message": f"saved {label}",
        }
    )


def finish_trigger_training_state_locked(session_id: str, state: dict[str, Any], ended_at: float) -> None:
    label = safe_trigger_label(state.get("label"))
    started_at = float(state.get("startedAt") or 0.0)
    samples = training_samples_for_window(session_id, started_at, max(started_at, ended_at))
    saved = save_trigger_training_samples(session_id, label, samples)
    state.update(
        {
            "inputState": "finished",
            "isRecording": False,
            "endsAt": ended_at,
            "serverSamples": len(samples),
            "savedPath": saved["savedPath"],
            "sampleCount": saved["sampleCount"],
            "counts": saved["counts"],
            "message": f"saved trigger {label}",
        }
    )


def refresh_shape_state_locked(session_id: str, current_time: float | None = None) -> None:
    samples = sorted(SESSION_SAMPLES.get(session_id, []), key=lambda sample: sample.get("time_ms", sample["received_at"]))
    if not samples:
        return
    latest_time = current_time if current_time is not None else samples[-1].get("time_ms", samples[-1]["received_at"])

    state = ACTIVE_INPUTS.setdefault(session_id, make_waiting_state(session_id))
    if state.get("inputState") == "finished":
        return
    state["serverSamples"] = len(training_samples_for_window(session_id, float(state.get("startedAt") or 0.0), latest_time))
    if state.get("controlMode") == "tap":
        state["latestMetrics"] = {
            "autoTriggerMode": "wand tap",
            "triggerModelStatus": "not used in tap mode",
            "message": "tap wand to start or finish",
        }
        if state.get("inputState") == "recording_motion" and latest_time >= float(state.get("endsAt") or 0.0):
            finish_recording_state_locked(session_id, state, float(state.get("endsAt") or latest_time))
        return
    armed_at = float(state.get("armedAt") or 0.0)
    if state.get("inputState") == "waiting_for_start" and latest_time < armed_at + TRIGGER_ARM_DELAY_MS:
        state["latestMetrics"] = {
            "triggerAxis": TRIGGER_AXIS,
            "rejectedReason": "trigger arm delay",
            "triggerGatePassed": False,
        }
    elif latest_time < float(state.get("triggerCooldownUntil") or 0.0):
        state["latestMetrics"] = {
            "triggerAxis": TRIGGER_AXIS,
            "rejectedReason": "trigger cooldown",
            "triggerGatePassed": False,
        }
    else:
        is_trigger, metrics = trigger_metrics(samples, latest_time)
        candidate_samples = metrics.pop("__candidateSamples", None)
        if candidate_samples:
            state["lastTriggerCandidate"] = candidate_samples
            state["lastTriggerCandidateAt"] = latest_time
            LAST_TRIGGER_CANDIDATES[session_id] = {
                "samples": list(candidate_samples),
                "capturedAt": latest_time,
                "metrics": dict(metrics),
            }
        state["latestMetrics"] = metrics
        if state.get("inputState") == "waiting_for_start" and is_trigger:
            start_at = latest_time + 180.0
            state.update(
                {
                    "inputState": "recording_motion",
                    "isRecording": True,
                    "startedAt": start_at,
                    "endsAt": start_at + MAX_COMMAND_SECONDS * 1000.0,
                    "serverSamples": 0,
                    "recognizedShape": "",
                    "confidence": 0.0,
                    "selectedSkill": choose_shape_skill(None),
                    "currentBestSkill": choose_shape_skill(None),
                    "triggerCooldownUntil": latest_time + THRUST_COOLDOWN_MS,
                    "ignoreStopUntil": latest_time + START_STOP_COOLDOWN_MS,
                    "message": "recording started",
                }
            )
        elif state.get("inputState") == "recording_motion":
            if latest_time >= float(state.get("endsAt") or 0.0):
                finish_recording_state_locked(session_id, state, float(state.get("endsAt") or latest_time))
            elif is_trigger and latest_time >= float(state.get("ignoreStopUntil") or 0.0):
                finish_recording_state_locked(session_id, state, latest_time - 180.0)
                state["triggerCooldownUntil"] = latest_time + THRUST_COOLDOWN_MS


def motion_status_payload(session_id: str) -> dict[str, Any]:
    with STATE_LOCK:
        if session_id in SESSION_SAMPLES:
            refresh_shape_state_locked(session_id)
        state = ACTIVE_INPUTS.setdefault(session_id, make_waiting_state(session_id))
        now_value = now_ms()
        remaining = max(0.0, (float(state.get("endsAt") or now_value) - now_value) / 1000.0)
        return {
            "ok": True,
            "sessionId": session_id,
            "controlMode": state.get("controlMode") or MOTION_INPUT_MODES.get(session_id, "trigger"),
            "inputState": state.get("inputState") or "waiting_for_start",
            "isRecording": bool(state.get("isRecording")),
            "startedAt": state.get("startedAt"),
            "endTime": state.get("endsAt"),
            "remainingSeconds": remaining,
            "serverSamples": int(state.get("serverSamples") or 0),
            "recognizedShape": state.get("recognizedShape") or "",
            "confidence": float(state.get("confidence") or 0.0),
            "selectedSkill": state.get("selectedSkill") or choose_shape_skill(None),
            "currentBestSkill": state.get("currentBestSkill") or choose_shape_skill(None),
            "latestMetrics": state.get("latestMetrics") or {},
            "message": state.get("message") or "",
        }


def append_memory_samples(session_id: str, samples: list[Any], received_at: float) -> int:
    cleaned: list[dict[str, float]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        x = coerce_float(sample.get("x"))
        y = coerce_float(sample.get("y"))
        z = coerce_float(sample.get("z"))
        magnitude = sample.get("magnitude")
        mag_value = coerce_float(magnitude) if magnitude is not None else math.sqrt(x * x + y * y + z * z)
        timestamp = coerce_float(sample.get("timestamp"))
        cleaned.append(
            {
                "timestamp": timestamp,
                "time_ms": sample_time_ms(timestamp, received_at),
                "received_at": received_at,
                "x": x,
                "y": y,
                "z": z,
                "magnitude": mag_value,
            }
        )

    if not cleaned:
        return 0

    with STATE_LOCK:
        existing = SESSION_SAMPLES.setdefault(session_id, [])
        existing.extend(cleaned)
        if len(existing) > MAX_SESSION_SAMPLES:
            del existing[: len(existing) - MAX_SESSION_SAMPLES]
        refresh_shape_state_locked(session_id, max(sample["time_ms"] for sample in cleaned))
    return len(cleaned)


def value_range(samples: list[dict[str, float]], key: str) -> float:
    values = [sample[key] for sample in samples]
    return max(values) - min(values)


def max_abs_diff(samples: list[dict[str, float]], key: str) -> float:
    if len(samples) < 2:
        return 0.0
    return max(abs(samples[idx][key] - samples[idx - 1][key]) for idx in range(1, len(samples)))


def predict_shape_label(window: list[dict[str, float]]) -> tuple[str | None, float | None, str]:
    if not SHAPE_MODEL_ARTIFACT:
        return None, None, SHAPE_MODEL_STATUS
    try:
        from scripts.extract_motion_features import extract_features_from_samples, features_to_vector

        model = SHAPE_MODEL_ARTIFACT["model"]
        features = extract_features_from_samples(window)
        vector = [features_to_vector(features)]
        label = str(model.predict(vector)[0])
        confidence: float | None = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(vector)[0]
            classes = [str(item) for item in getattr(model, "classes_", SHAPE_MODEL_ARTIFACT.get("labels", []))]
            active_scores = [
                (classes[index], float(probability))
                for index, probability in enumerate(probabilities)
                if index < len(classes) and classes[index] in ACTIVE_SHAPE_LABELS
            ]
            if active_scores:
                label, confidence = max(active_scores, key=lambda item: item[1])
                return label, confidence, "enabled active-3"
            confidence = float(max(probabilities))
        if label not in ACTIVE_SHAPE_LABELS:
            return "circle", confidence, "enabled active-3 fallback"
        return label, confidence, "enabled active-3"
    except Exception as exc:
        return None, None, f"model error: {exc}"


def detect_motion_sequence(samples: list[dict[str, float]]) -> dict[str, Any]:
    if len(samples) < 3:
        return {
            "sequence": ["U"],
            "segments": [],
            "threshold": 0.0,
            "baseline": 0.0,
            "message": "not enough samples",
        }

    ordered = sorted(samples, key=lambda sample: sample.get("time_ms", sample["received_at"]))
    first_time = ordered[0].get("time_ms", ordered[0]["received_at"])
    baseline_samples = [sample for sample in ordered if sample.get("time_ms", sample["received_at"]) - first_time <= 1000.0]
    if not baseline_samples:
        baseline_samples = ordered[: min(len(ordered), 20)]

    baseline = sum(sample["magnitude"] for sample in baseline_samples) / len(baseline_samples)
    deltas = [abs(sample["magnitude"] - baseline) for sample in ordered]
    max_delta = max(deltas)
    threshold = max(1.2, min(3.8, max_delta * 0.5))

    if max_delta < 1.2:
        return {
            "sequence": ["U"],
            "segments": [],
            "threshold": threshold,
            "baseline": baseline,
            "message": "motion was too small",
        }

    raw_segments: list[tuple[int, int]] = []
    in_motion = False
    start_idx = 0
    quiet_since: float | None = None
    min_duration_ms = 100.0
    quiet_duration_ms = 240.0

    for idx, sample in enumerate(ordered):
        active = abs(sample["magnitude"] - baseline) >= threshold
        current_time = sample.get("time_ms", sample["received_at"])

        if active and not in_motion:
            in_motion = True
            start_idx = idx
            quiet_since = None
        elif active and in_motion:
            quiet_since = None
        elif in_motion:
            if quiet_since is None:
                quiet_since = current_time
            if current_time - quiet_since >= quiet_duration_ms:
                end_idx = idx
                if (
                    ordered[end_idx].get("time_ms", ordered[end_idx]["received_at"])
                    - ordered[start_idx].get("time_ms", ordered[start_idx]["received_at"])
                    >= min_duration_ms
                ):
                    raw_segments.append((start_idx, end_idx))
                in_motion = False
                quiet_since = None

    if in_motion:
        end_idx = len(ordered) - 1
        if (
            ordered[end_idx].get("time_ms", ordered[end_idx]["received_at"])
            - ordered[start_idx].get("time_ms", ordered[start_idx]["received_at"])
            >= min_duration_ms
        ):
            raw_segments.append((start_idx, end_idx))

    if not raw_segments:
        return {
            "sequence": ["U"],
            "segments": [],
            "threshold": threshold,
            "baseline": baseline,
            "message": "no clear shake segment",
        }

    sequence: list[str] = []
    segments: list[dict[str, Any]] = []
    for segment_id, (start_idx, end_idx) in enumerate(raw_segments[:12], start=1):
        window = ordered[start_idx : end_idx + 1]
        ranges = {
            "x": value_range(window, "x"),
            "y": value_range(window, "y"),
            "z": value_range(window, "z"),
        }
        code, metrics = command_from_window(window)
        if code == "U":
            continue
        dominant_axis = str(metrics["bestAxis"])
        peak_magnitude = max(sample["magnitude"] for sample in window)
        peak_delta = max(abs(sample["magnitude"] - baseline) for sample in window)
        start_time_ms = ordered[start_idx].get("time_ms", ordered[start_idx]["received_at"])
        end_time_ms = ordered[end_idx].get("time_ms", ordered[end_idx]["received_at"])
        duration = (end_time_ms - start_time_ms) / 1000.0
        strength = "strong" if peak_delta >= max(2.5, threshold * 3.0) else "normal"
        repeats = 1
        sequence.append(code)
        segments.append(
            {
                "segment_id": segment_id,
                "start_time": (start_time_ms - first_time) / 1000.0,
                "end_time": (end_time_ms - first_time) / 1000.0,
                "duration": duration,
                "peak_magnitude": peak_magnitude,
                "peak_delta": peak_delta,
                "dominant_axis": dominant_axis,
                "code": code,
                "strength": strength,
                "repeats": repeats,
                "dominance_ratio": metrics["bestScore"] / max(float(metrics["secondScore"]), 0.001),
            }
        )

    if not sequence:
        sequence = ["U"]

    return {
        "sequence": sequence[:12],
        "segments": segments,
        "threshold": threshold,
        "baseline": baseline,
        "message": "ok",
    }


class MotionLogHandler(BaseHTTPRequestHandler):
    server_version = "MotionLogHTTP/0.1"

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS, GET")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/health":
            self.send_json(200, {"status": "ok"})
            return
        if path == "/api/training/status":
            query = parse_qs(parsed.query)
            session_id = safe_session_id((query.get("sessionId") or ["battle-room"])[0])
            with STATE_LOCK:
                if session_id in SESSION_SAMPLES:
                    refresh_shape_state_locked(session_id)
                active_training = dict(ACTIVE_TRAINING.get(session_id) or {})
            self.send_json(
                200,
                {
                    "ok": True,
                    "counts": training_counts(),
                    "triggerCounts": trigger_training_counts(),
                    "active": active_training,
                    **model_status_payload(),
                },
            )
            return
        if path in {"/api/game/input/status", "/api/motion/status"}:
            query = parse_qs(parsed.query)
            session_id = safe_session_id((query.get("sessionId") or ["battle-room"])[0])
            payload = motion_status_payload(session_id)
            payload["sampleCount"] = payload["serverSamples"]
            payload["active"] = ACTIVE_INPUTS.get(session_id)
            payload.update(model_status_payload())
            self.send_json(200, payload)
            return
        if path.startswith("/api/"):
            self.send_json(404, {"error": "not found"})
            return
        self.send_static_file(parsed.path)
        return

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path in {"/api/logs", "/api/logs/batch"}:
            self.handle_logs()
            return
        if path in {"/api/game/input/start", "/api/motion/start"}:
            self.handle_game_input_start()
            return
        if path == "/api/motion/mode":
            self.handle_motion_input_mode()
            return
        if path == "/api/motion/tap":
            self.handle_motion_wand_tap()
            return
        if path in {"/api/game/input/result", "/api/motion/finish"}:
            self.handle_game_input_result()
            return
        if path in {"/api/game/input/cancel", "/api/motion/cancel"}:
            self.handle_game_input_cancel()
            return
        if path == "/api/training/record/start":
            self.handle_training_record_start()
            return
        if path == "/api/training/record/finish":
            self.handle_training_record_finish()
            return
        if path == "/api/training/record/cancel":
            self.handle_training_record_cancel()
            return
        if path in {"/api/training/trigger/start", "/api/training/trigger/record/start", "/api/trigger/start"}:
            self.handle_trigger_record_start()
            return
        if path in {"/api/training/trigger/finish", "/api/training/trigger/record/finish", "/api/trigger/finish"}:
            self.handle_trigger_record_finish()
            return
        if path in {"/api/training/trigger/save-candidate", "/api/trigger/save-candidate"}:
            self.handle_trigger_candidate_save()
            return
        if path in {"/api/training/shape/save-last", "/api/shape/save-last"}:
            self.handle_shape_candidate_save()
            return
        if path == "/api/training/train":
            self.handle_training_train()
            return
        if path in {"/api/training/trigger/train", "/api/trigger/train"}:
            self.handle_trigger_train()
            return
        if path in {"/api/training/trigger/archive", "/api/trigger/archive"}:
            self.handle_trigger_archive()
            return
        self.send_json(404, {"error": "not found", "path": path, "method": "POST"})

    def read_json_payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        return payload

    def handle_logs(self) -> None:
        try:
            payload = self.read_json_payload()
            samples = payload.get("samples", [])
            if not isinstance(samples, list):
                raise ValueError("samples must be a list")

            session_id = safe_session_id(payload.get("sessionId"))
            received_at = now_ms()
            memory_saved = append_memory_samples(session_id, samples, received_at)
            path = csv_path_for_session(session_id)
            is_new_file = not path.exists()

            with path.open("a", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
                if is_new_file:
                    writer.writeheader()
                for sample in samples:
                    writer.writerow(
                        {
                            "timestamp": coerce_float(sample.get("timestamp")),
                            "x": coerce_float(sample.get("x")),
                            "y": coerce_float(sample.get("y")),
                            "z": coerce_float(sample.get("z")),
                            "magnitude": coerce_float(sample.get("magnitude")),
                        }
                    )

            print(f"[{datetime.now().isoformat(timespec='seconds')}] saved {len(samples)} samples -> {path}")
            self.send_json(
                200,
                {
                    "ok": True,
                    "saved": len(samples),
                    "memorySaved": memory_saved,
                    "file": str(path),
                },
            )
        except Exception as exc:
            print(f"[ERROR] {exc}")
            self.send_json(400, {"ok": False, "error": str(exc)})

    def handle_game_input_start(self) -> None:
        try:
            payload = self.read_json_payload()
            session_id = safe_session_id(payload.get("sessionId") or "battle-room")
            with STATE_LOCK:
                ACTIVE_INPUTS[session_id] = make_waiting_state(session_id)
            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] game input started "
                f"session={session_id} waiting_for_start"
            )
            self.send_json(
                200,
                {
                    "ok": True,
                    "sessionId": session_id,
                    "controlMode": MOTION_INPUT_MODES.get(session_id, "trigger"),
                    "inputState": "waiting_for_start",
                    "message": "tap the wand to start casting" if MOTION_INPUT_MODES.get(session_id) == "tap" else "move phone forward to start casting",
                },
            )
        except Exception as exc:
            print(f"[ERROR] {exc}")
            self.send_json(400, {"ok": False, "error": str(exc)})

    def handle_motion_input_mode(self) -> None:
        try:
            payload = self.read_json_payload()
            session_id = safe_session_id(payload.get("sessionId") or "battle-room")
            control_mode = safe_motion_input_mode(payload.get("mode"))
            with STATE_LOCK:
                active = ACTIVE_INPUTS.get(session_id)
                if active and active.get("inputState") == "recording_motion":
                    raise ValueError("cannot change input mode while casting")
                MOTION_INPUT_MODES[session_id] = control_mode
                if active:
                    active["controlMode"] = control_mode
                    active["latestMetrics"] = {
                        "autoTriggerMode": "wand tap" if control_mode == "tap" else TRIGGER_MODE,
                        "message": "input mode changed",
                    }
            self.send_json(200, {
                "ok": True,
                "sessionId": session_id,
                "controlMode": control_mode,
                "message": f"input mode changed to {control_mode}",
            })
        except Exception as exc:
            print(f"[ERROR] {exc}")
            self.send_json(409, {"ok": False, "error": str(exc)})

    def handle_motion_wand_tap(self) -> None:
        try:
            payload = self.read_json_payload()
            session_id = safe_session_id(payload.get("sessionId") or "battle-room")
            tapped_at = now_ms()
            with STATE_LOCK:
                if MOTION_INPUT_MODES.get(session_id, "trigger") != "tap":
                    raise ValueError("wand tap mode is not enabled")
                active = ACTIVE_INPUTS.setdefault(session_id, make_waiting_state(session_id))
                active["controlMode"] = "tap"
                input_state = active.get("inputState")
                if input_state == "waiting_for_start":
                    start_at = tapped_at + 100.0
                    active.update({
                        "inputState": "recording_motion",
                        "isRecording": True,
                        "startedAt": start_at,
                        "endsAt": start_at + MAX_COMMAND_SECONDS * 1000.0,
                        "serverSamples": 0,
                        "recognizedShape": "",
                        "confidence": 0.0,
                        "selectedSkill": choose_shape_skill(None),
                        "currentBestSkill": choose_shape_skill(None),
                        "latestMetrics": {"autoTriggerMode": "wand tap", "message": "recording started by wand tap"},
                        "message": "recording started by wand tap",
                    })
                elif input_state == "recording_motion":
                    started_at = float(active.get("startedAt") or tapped_at)
                    if tapped_at - started_at < TAP_MIN_RECORDING_MS:
                        active["latestMetrics"] = {
                            "autoTriggerMode": "wand tap",
                            "message": "tap ignored: recording just started",
                        }
                        active["message"] = "keep drawing before the finish tap"
                    else:
                        finish_at = max(started_at, tapped_at - 100.0)
                        finish_recording_state_locked(session_id, active, finish_at)
                        active["controlMode"] = "tap"
                elif input_state == "finished":
                    raise ValueError("cast is already finished")
                else:
                    raise ValueError(f"wand tap is not available in state {input_state}")
                response = {
                    "ok": True,
                    "sessionId": session_id,
                    "controlMode": "tap",
                    "inputState": active.get("inputState"),
                    "isRecording": bool(active.get("isRecording")),
                    "recognizedShape": active.get("recognizedShape") or "",
                    "confidence": float(active.get("confidence") or 0.0),
                    "serverSamples": int(active.get("serverSamples") or 0),
                    "message": active.get("message") or "",
                }
            self.send_json(200, response)
        except Exception as exc:
            print(f"[ERROR] {exc}")
            self.send_json(409, {"ok": False, "error": str(exc)})

    def handle_game_input_result(self) -> None:
        try:
            payload = self.read_json_payload()
            session_id = safe_session_id(payload.get("sessionId") or "battle-room")
            ended_at = now_ms()
            with STATE_LOCK:
                active = ACTIVE_INPUTS.get(session_id)
                if not active:
                    active = make_waiting_state(session_id)
                    ACTIVE_INPUTS[session_id] = active
                if active.get("inputState") == "recording_motion":
                    finish_recording_state_locked(session_id, active, ended_at)
                selected_skill = active.get("selectedSkill") or choose_shape_skill(active.get("recognizedShape"))
                recognized_shape = active.get("recognizedShape") or "none"
                sample_count = int(active.get("serverSamples") or 0)
            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] game input result "
                f"session={session_id} samples={sample_count} shape={recognized_shape}"
            )
            self.send_json(
                200,
                {
                    "ok": True,
                    "sessionId": session_id,
                    "inputState": active.get("inputState"),
                    "sampleCount": sample_count,
                    "serverSamples": sample_count,
                    "recognizedShape": recognized_shape,
                    "confidence": float(active.get("confidence") or 0.0),
                    "selectedSkill": selected_skill,
                    "currentBestSkill": selected_skill,
                    "message": active.get("message") or "",
                },
            )
        except Exception as exc:
            print(f"[ERROR] {exc}")
            self.send_json(400, {"ok": False, "error": str(exc)})

    def handle_game_input_cancel(self) -> None:
        try:
            payload = self.read_json_payload()
            session_id = safe_session_id(payload.get("sessionId") or "battle-room")
            with STATE_LOCK:
                active = ACTIVE_INPUTS.pop(session_id, None)
            self.send_json(
                200,
                {
                    "ok": True,
                    "sessionId": session_id,
                    "cancelled": bool(active),
                    "message": "motion input cancelled",
                },
            )
        except Exception as exc:
            print(f"[ERROR] {exc}")
            self.send_json(400, {"ok": False, "error": str(exc)})

    def handle_training_record_start(self) -> None:
        try:
            payload = self.read_json_payload()
            session_id = safe_session_id(payload.get("sessionId") or "battle-room")
            label = safe_training_label(payload.get("label"))
            started_at = now_ms()
            with STATE_LOCK:
                ACTIVE_TRAINING[session_id] = {
                    "sessionId": session_id,
                    "label": label,
                    "kind": "shape",
                    "inputState": "recording_motion",
                    "isRecording": True,
                    "startedAt": started_at,
                    "endsAt": started_at + MAX_COMMAND_SECONDS * 1000.0,
                    "serverSamples": 0,
                    "message": "recording shape motion",
                }
            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] training record started "
                f"session={session_id} label={label} recording_motion"
            )
            self.send_json(
                200,
                {
                    "ok": True,
                    "sessionId": session_id,
                    "label": label,
                    "inputState": "recording_motion",
                    "startedAt": started_at,
                    "message": "draw the shape, then press Stop",
                },
            )
        except Exception as exc:
            print(f"[ERROR] {exc}")
            self.send_json(400, {"ok": False, "error": str(exc)})

    def handle_training_record_finish(self) -> None:
        try:
            payload = self.read_json_payload()
            session_id = safe_session_id(payload.get("sessionId") or "battle-room")
            label = safe_training_label(payload.get("label"))
            ended_at = now_ms()
            with STATE_LOCK:
                active = ACTIVE_TRAINING.get(session_id)
                if not active:
                    raise ValueError("no active training record")
                if active.get("kind") != "shape":
                    raise ValueError("active training record is not shape training")
                if active.get("inputState") == "recording_motion":
                    active["label"] = label
                    finish_training_state_locked(session_id, active, ended_at)
                if active.get("inputState") != "finished":
                    raise ValueError("recording is not finished yet. press Stop to save.")
                sample_count = int(active.get("sampleCount") or 0)
                saved_path = str(active.get("savedPath") or "")

            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] training saved "
                f"label={label} samples={sample_count} -> {saved_path}"
            )
            self.send_json(
                200,
                {
                    "ok": True,
                    "sessionId": session_id,
                    "label": label,
                    "savedPath": saved_path,
                    "sampleCount": sample_count,
                    "counts": training_counts(),
                    **model_status_payload(),
                },
            )
        except Exception as exc:
            print(f"[ERROR] {exc}")
            self.send_json(400, {"ok": False, "error": str(exc)})

    def handle_training_record_cancel(self) -> None:
        try:
            payload = self.read_json_payload()
            session_id = safe_session_id(payload.get("sessionId") or "battle-room")
            with STATE_LOCK:
                active = ACTIVE_TRAINING.pop(session_id, None)
            self.send_json(
                200,
                {
                    "ok": True,
                    "sessionId": session_id,
                    "cancelled": bool(active),
                    "message": "training record discarded",
                },
            )
        except Exception as exc:
            print(f"[ERROR] {exc}")
            self.send_json(400, {"ok": False, "error": str(exc)})

    def handle_training_train(self) -> None:
        try:
            from scripts.train_motion_model import train_model

            result = train_model(TRAINING_DIR, SHAPE_MODEL_PATH)
            load_motion_model()
            result.update(model_status_payload())
            status = 200 if result.get("ok") else 400
            self.send_json(status, result)
        except Exception as exc:
            print(f"[ERROR] {exc}")
            self.send_json(400, {"ok": False, "error": str(exc), **model_status_payload()})

    def handle_trigger_record_start(self) -> None:
        try:
            payload = self.read_json_payload()
            session_id = safe_session_id(payload.get("sessionId") or "battle-room")
            label = safe_trigger_label(payload.get("label"))
            started_at = now_ms()
            key = f"{session_id}:trigger"
            with STATE_LOCK:
                ACTIVE_TRAINING[key] = {
                    "sessionId": session_id,
                    "label": label,
                    "kind": "trigger",
                    "inputState": "recording_motion",
                    "isRecording": True,
                    "startedAt": started_at,
                    "endsAt": started_at + MAX_COMMAND_SECONDS * 1000.0,
                    "serverSamples": 0,
                    "message": "recording trigger sample",
                }
            self.send_json(
                200,
                {
                    "ok": True,
                    "sessionId": session_id,
                    "label": label,
                    "inputState": "recording_motion",
                    "startedAt": started_at,
                    "message": "perform the trigger gesture, then press Stop",
                },
            )
        except Exception as exc:
            print(f"[ERROR] {exc}")
            self.send_json(400, {"ok": False, "error": str(exc)})

    def handle_trigger_record_finish(self) -> None:
        try:
            payload = self.read_json_payload()
            session_id = safe_session_id(payload.get("sessionId") or "battle-room")
            label = safe_trigger_label(payload.get("label"))
            ended_at = now_ms()
            key = f"{session_id}:trigger"
            with STATE_LOCK:
                active = ACTIVE_TRAINING.get(key)
                if not active:
                    raise ValueError("no active trigger training record")
                if active.get("kind") != "trigger":
                    raise ValueError("active training record is not trigger training")
                if active.get("label") != label:
                    raise ValueError("label does not match active trigger training record")
                finish_trigger_training_state_locked(session_id, active, ended_at)
                sample_count = int(active.get("sampleCount") or 0)
                saved_path = str(active.get("savedPath") or "")
            self.send_json(
                200,
                {
                    "ok": True,
                    "sessionId": session_id,
                    "label": label,
                    "savedPath": saved_path,
                    "sampleCount": sample_count,
                    "triggerCounts": trigger_training_counts(),
                    **model_status_payload(),
                },
            )
        except Exception as exc:
            print(f"[ERROR] {exc}")
            self.send_json(400, {"ok": False, "error": str(exc)})

    def handle_trigger_candidate_save(self) -> None:
        try:
            payload = self.read_json_payload()
            session_id = safe_session_id(payload.get("sessionId") or "battle-room")
            label = safe_trigger_label(payload.get("label"))
            with STATE_LOCK:
                active = ACTIVE_INPUTS.get(session_id)
                samples: list[dict[str, float]] = []
                if active:
                    samples = list(active.get("lastTriggerCandidate") or [])
                if not samples:
                    cached = LAST_TRIGGER_CANDIDATES.get(session_id) or {}
                    samples = list(cached.get("samples") or [])
                if not samples:
                    session_samples = sorted(
                        SESSION_SAMPLES.get(session_id, []),
                        key=lambda sample: sample.get("time_ms", sample["received_at"]),
                    )
                    if session_samples:
                        latest_time = session_samples[-1].get("time_ms", session_samples[-1]["received_at"])
                        start_time = latest_time - max(TRIGGER_WINDOW_MS, THRUST_WINDOW_MS)
                        samples = [
                            sample
                            for sample in session_samples
                            if start_time <= sample.get("time_ms", sample["received_at"]) <= latest_time
                        ]
                if not samples:
                    raise ValueError("no trigger candidate or recent samples available")
                saved = save_trigger_training_samples(session_id, label, samples)
            self.send_json(
                200,
                {
                    "ok": True,
                    "sessionId": session_id,
                    "label": label,
                    "message": f"saved last trigger candidate as {label}",
                    "savedPath": saved["savedPath"],
                    "sampleCount": saved["sampleCount"],
                    "triggerCounts": saved["counts"],
                    **model_status_payload(),
                },
            )
        except Exception as exc:
            print(f"[ERROR] {exc}")
            self.send_json(400, {"ok": False, "error": str(exc), **model_status_payload()})

    def handle_shape_candidate_save(self) -> None:
        try:
            payload = self.read_json_payload()
            session_id = safe_session_id(payload.get("sessionId") or "battle-room")
            label = safe_training_label(payload.get("label"))
            with STATE_LOCK:
                active = ACTIVE_INPUTS.get(session_id)
                samples: list[dict[str, float]] = []
                if active:
                    samples = list(active.get("lastShapeSamples") or [])
                if not samples:
                    cached = LAST_SHAPE_CANDIDATES.get(session_id) or {}
                    samples = list(cached.get("samples") or [])
                if not samples:
                    raise ValueError("no finished shape input samples available")
                saved = save_training_samples(session_id, label, samples)
            self.send_json(
                200,
                {
                    "ok": True,
                    "sessionId": session_id,
                    "label": label,
                    "message": f"saved last shape input as {label}",
                    "savedPath": saved["savedPath"],
                    "sampleCount": saved["sampleCount"],
                    "counts": saved["counts"],
                    **model_status_payload(),
                },
            )
        except Exception as exc:
            print(f"[ERROR] {exc}")
            self.send_json(400, {"ok": False, "error": str(exc), **model_status_payload()})

    def handle_trigger_train(self) -> None:
        try:
            from scripts.train_trigger_model import train_trigger_model

            result = train_trigger_model(
                TRIGGER_TRAINING_DIR,
                TRAINING_DIR,
                TRIGGER_MODEL_PATH,
                use_legacy_p_data=False,
            )
            load_trigger_model()
            status_payload = model_status_payload()
            result.update(
                {
                    key: value
                    for key, value in status_payload.items()
                    if key not in {"modelExists", "modelFileExists", "modelPath", "modelStatus", "modelLabels"}
                }
            )
            status = 200 if result.get("ok") else 400
            self.send_json(status, result)
        except Exception as exc:
            print(f"[ERROR] {exc}")
            self.send_json(400, {"ok": False, "error": str(exc), **model_status_payload()})

    def handle_trigger_archive(self) -> None:
        try:
            payload = self.read_json_payload()
            if payload.get("confirm") is not True:
                raise ValueError("confirm must be true")
            with STATE_LOCK:
                result = archive_trigger_training_data()
                LAST_TRIGGER_CANDIDATES.clear()
                for state in ACTIVE_INPUTS.values():
                    state["lastTriggerCandidate"] = []
                    state["lastTriggerCandidateAt"] = 0.0
                load_trigger_model()
            self.send_json(
                200,
                {
                    "ok": True,
                    "message": "archived trigger training data and model",
                    "triggerCounts": trigger_training_counts(),
                    **result,
                    **model_status_payload(),
                },
            )
        except Exception as exc:
            print(f"[ERROR] {exc}")
            self.send_json(400, {"ok": False, "error": str(exc), **model_status_payload()})

    def send_json(self, status: int, data: dict[str, Any]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_static_file(self, request_path: str) -> None:
        if not FRONTEND_DIST_DIR.exists():
            self.send_json(
                404,
                {
                    "error": "frontend build not found",
                    "message": "Run npm.cmd run build in frontend first.",
                    "dist": str(FRONTEND_DIST_DIR),
                },
            )
            return

        decoded_path = unquote(request_path.split("?", 1)[0])
        if decoded_path in {"", "/"}:
            relative_path = Path("index.html")
        else:
            relative_path = Path(decoded_path.lstrip("/"))

        candidate = (FRONTEND_DIST_DIR / relative_path).resolve()
        dist_root = FRONTEND_DIST_DIR.resolve()
        try:
            candidate.relative_to(dist_root)
        except ValueError:
            self.send_json(403, {"error": "forbidden"})
            return

        if not candidate.exists() or candidate.is_dir():
            candidate = dist_root / "index.html"

        try:
            body = candidate.read_bytes()
        except OSError as exc:
            self.send_json(500, {"error": f"failed to read static file: {exc}"})
            return

        suffix = candidate.suffix.lower()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if suffix == ".js":
            content_type = "text/javascript"
        elif suffix == ".png":
            content_type = "image/png"
        elif suffix in {".jpg", ".jpeg"}:
            content_type = "image/jpeg"
        elif suffix == ".webp":
            content_type = "image/webp"
        elif suffix == ".svg":
            content_type = "image/svg+xml"

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    host = "0.0.0.0"
    port = 8000
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    TRIGGER_TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    load_motion_model()
    server = ThreadingHTTPServer((host, port), MotionLogHandler)
    print(f"Motion log server running on http://{host}:{port}")
    print(f"CSV output directory: {RAW_DIR}")
    print(f"Training output directory: {TRAINING_DIR}")
    print(f"Shape model: {SHAPE_MODEL_STATUS} ({SHAPE_MODEL_PATH})")
    print(f"Trigger model: {TRIGGER_MODEL_STATUS} ({TRIGGER_MODEL_PATH})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
