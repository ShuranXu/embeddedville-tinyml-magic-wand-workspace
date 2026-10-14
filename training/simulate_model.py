"""Run the exported int8 model against recorder data without a physical board.

This uses the TensorFlow Lite host interpreter to validate the same raster bytes,
tensor metadata, labels, thresholds, and exported model used by the firmware. It
does not emulate the IMU, BLE radio, nRF52840 timing, or tensor-arena allocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from train_magic_wand import LABELS, load_dataset, tflite_predictions


def classify(scores: np.ndarray, confidence: float, margin: float) -> tuple[str, float, float, str]:
    order = np.argsort(scores)[::-1]
    top, runner_up = int(order[0]), int(order[1])
    top_score = float(scores[top])
    score_margin = top_score - float(scores[runner_up])
    accepted = LABELS[top] != "unknown" and top_score >= confidence and score_margin >= margin
    return (LABELS[top] if accepted else "unknown", top_score, score_margin, LABELS[top])


def simulate(dataset_path: Path, model_path: Path, split: str = "test", confidence: float = 0.65, margin: float = 0.15, *, allow_fixture: bool = False) -> dict:
    images, label_ids = load_dataset(dataset_path, allow_fixture=allow_fixture)
    model_bytes = model_path.read_bytes()
    scores, metadata, operators = tflite_predictions(model_bytes, images[split])
    trials = []
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for index, (actual_id, output) in enumerate(zip(label_ids[split], scores, strict=True), start=1):
        prediction, top_score, score_margin, top_label = classify(output, confidence, margin)
        actual = LABELS[int(actual_id)]
        matrix[int(actual_id)][LABELS.index(prediction)] += 1
        trials.append({"number": index, "actualLabel": actual, "predictedLabel": prediction, "topLabel": top_label, "confidence": top_score, "margin": score_margin, "scores": [float(value) for value in output]})
    correct = sum(trial["actualLabel"] == trial["predictedLabel"] for trial in trials)
    unknown = [trial for trial in trials if trial["actualLabel"] == "unknown"]
    false_accepts = sum(trial["predictedLabel"] != "unknown" for trial in unknown)
    return {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "runtime": "TensorFlow Lite host interpreter",
        "model": {"path": model_path.name, "bytes": len(model_bytes), "sha256": hashlib.sha256(model_bytes).hexdigest(), "operators": sorted(operators), "tensorMetadata": metadata},
        "dataset": {"path": dataset_path.name, "split": split, "examples": len(trials)},
        "thresholds": {"confidence": confidence, "margin": margin},
        "summary": {"accuracy": correct / max(1, len(trials)), "unknownFalseAcceptance": false_accepts / max(1, len(unknown)), "confusionMatrix": matrix},
        "trials": trials,
        "evidenceBoundary": "Required virtual-device preprocessing, exported-model inference, thresholds, and dataset replay",
        "limitations": ["No physical IMU noise or orientation behavior", "No BLE transport timing", "No nRF52840 cycle, tensor-arena, or power measurement", "Physical-device testing is an optional extension and is not required for certification"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a Magic Wand dataset through the exported int8 model")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--confidence", type=float, default=0.65)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-fixture", action="store_true", help="Maintainer-only: replay the bundled non-submittable fixture")
    args = parser.parse_args()
    report = simulate(args.dataset, args.model, args.split, args.confidence, args.margin, allow_fixture=args.allow_fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
