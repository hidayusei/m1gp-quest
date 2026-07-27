from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from extract_motion_features import FEATURE_COLUMNS, collect_training_features
except ImportError:
    from scripts.extract_motion_features import FEATURE_COLUMNS, collect_training_features


def train_model(training_dir: Path, model_path: Path) -> dict[str, Any]:
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

    if not training_dir.exists():
        return {
            "ok": False,
            "message": f"training directory not found: {training_dir}",
            "counts": {},
            "modelPath": str(model_path),
        }

    vectors, labels, rows = collect_training_features(training_dir)
    counts = dict(Counter(labels))
    if len(vectors) < 2 or len(counts) < 2:
        return {
            "ok": False,
            "message": "need at least 2 shape labels and 2 samples to train",
            "counts": counts,
            "modelPath": str(model_path),
        }

    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=1,
    )

    simple_training = len(vectors) < 8 or min(counts.values()) < 2
    accuracy: float | None = None
    if simple_training:
        model.fit(vectors, labels)
        message = "model trained with all samples because dataset is small"
    else:
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                vectors,
                labels,
                test_size=0.25,
                random_state=42,
                stratify=labels,
            )
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            accuracy = float(accuracy_score(y_test, predictions))
            message = "model trained"
        except ValueError:
            model.fit(vectors, labels)
            message = "model trained with all samples because split was not possible"

    artifact = {
        "model": model,
        "featureColumns": FEATURE_COLUMNS,
        "labels": sorted(counts.keys()),
        "counts": counts,
        "sampleCount": len(rows),
        "message": message,
        "accuracy": accuracy,
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
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train a RandomForest circle/star/zigzag motion classifier.")
    parser.add_argument("--training-dir", default=str(base_dir / "data" / "training"))
    parser.add_argument("--model-path", default=str(base_dir / "models" / "shape_classifier.pkl"))
    args = parser.parse_args()

    result = train_model(Path(args.training_dir), Path(args.model_path))
    print(result["message"])
    print(f"model: {result['modelPath']}")
    print(f"counts: {result.get('counts', {})}")
    if result.get("accuracy") is not None:
        print(f"accuracy: {result['accuracy']:.3f}")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
