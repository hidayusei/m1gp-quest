from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from extract_motion_features import FEATURE_COLUMNS, features_to_vector, extract_features_from_samples, read_motion_csv
except ImportError:
    from scripts.extract_motion_features import FEATURE_COLUMNS, features_to_vector, extract_features_from_samples, read_motion_csv


def collect_trigger_features(
    trigger_training_dir: Path,
    legacy_training_dir: Path,
    use_legacy_p_data: bool = False,
) -> tuple[list[list[float]], list[str], list[dict[str, Any]]]:
    sources: list[tuple[str, Path]] = []
    for label in ("trigger", "none"):
        label_dir = trigger_training_dir / label
        if label_dir.exists():
            sources.extend((label, path) for path in sorted(label_dir.glob("*.csv")))

    if use_legacy_p_data:
        legacy_positive = legacy_training_dir / "P"
        if legacy_positive.exists():
            sources.extend(("trigger", path) for path in sorted(legacy_positive.glob("*.csv")))

        for legacy_label in ("H", "V", "none"):
            label_dir = legacy_training_dir / legacy_label
            if label_dir.exists():
                sources.extend(("none", path) for path in sorted(label_dir.glob("*.csv")))

    vectors: list[list[float]] = []
    labels: list[str] = []
    rows: list[dict[str, Any]] = []
    for label, csv_path in sources:
        samples = read_motion_csv(csv_path)
        if not samples:
            continue
        features = extract_features_from_samples(samples)
        vectors.append(features_to_vector(features))
        labels.append(label)
        rows.append({"label": label, "path": str(csv_path), **features})
    return vectors, labels, rows


def train_trigger_model(
    trigger_training_dir: Path,
    legacy_training_dir: Path,
    model_path: Path,
    use_legacy_p_data: bool = False,
) -> dict[str, Any]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        return {
            "ok": False,
            "message": "scikit-learn is not installed. Run: pip install -r requirements.txt",
            "error": str(exc),
            "counts": {},
            "modelPath": str(model_path),
        }

    vectors, labels, rows = collect_trigger_features(trigger_training_dir, legacy_training_dir, use_legacy_p_data)
    counts = dict(Counter(labels))
    if len(vectors) < 2 or len(counts) < 2:
        return {
            "ok": False,
            "message": "need trigger and none samples to train trigger detector",
            "counts": counts,
            "modelPath": str(model_path),
        }

    model = RandomForestClassifier(
        n_estimators=120,
        random_state=7,
        class_weight="balanced",
        min_samples_leaf=1,
    )

    simple_training = len(vectors) < 8 or min(counts.values()) < 2
    accuracy: float | None = None
    if simple_training:
        model.fit(vectors, labels)
        message = "trigger detector trained with all samples because dataset is small"
    else:
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                vectors,
                labels,
                test_size=0.25,
                random_state=7,
                stratify=labels,
            )
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            accuracy = float(accuracy_score(y_test, predictions))
            message = "trigger detector trained"
        except ValueError:
            model.fit(vectors, labels)
            message = "trigger detector trained with all samples because split was not possible"

    artifact = {
        "model": model,
        "featureColumns": FEATURE_COLUMNS,
        "labels": sorted(counts.keys()),
        "counts": counts,
        "sampleCount": len(rows),
        "message": message,
        "accuracy": accuracy,
        "kind": "trigger",
        "useLegacyPData": use_legacy_p_data,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as model_file:
        pickle.dump(artifact, model_file)

    return {
        "ok": True,
        "modelPath": str(model_path),
        "counts": counts,
        "sampleCount": len(rows),
        "accuracy": accuracy,
        "message": message,
        "useLegacyPData": use_legacy_p_data,
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train trigger/none detector for cast start/confirm gesture.")
    parser.add_argument("--trigger-training-dir", default=str(base_dir / "data" / "training_trigger"))
    parser.add_argument("--legacy-training-dir", default=str(base_dir / "data" / "training"))
    parser.add_argument("--model-path", default=str(base_dir / "models" / "trigger_detector.pkl"))
    parser.add_argument("--use-legacy-p-data", action="store_true")
    args = parser.parse_args()

    result = train_trigger_model(
        Path(args.trigger_training_dir),
        Path(args.legacy_training_dir),
        Path(args.model_path),
        use_legacy_p_data=args.use_legacy_p_data,
    )
    print(result["message"])
    print(f"model: {result['modelPath']}")
    print(f"counts: {result.get('counts', {})}")
    if result.get("accuracy") is not None:
        print(f"accuracy: {result['accuracy']:.3f}")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
