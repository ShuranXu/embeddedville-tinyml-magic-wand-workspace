from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

LABELS = ("wing", "ring", "slope", "unknown")
EXPECTED = {"train": 40, "validation": 10, "test": 10}


def validate_dataset(path: Path, *, allow_fixture: bool = False) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    schema = json.loads((Path(__file__).parent / "dataset.schema.json").read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(data), key=lambda error: list(error.path))
    if errors:
        raise ValueError("Schema validation failed:\n" + "\n".join(f"- {'/'.join(map(str, error.path)) or '<root>'}: {error.message}" for error in errors[:20]))
    if data["boardProfile"] == "fixture" and not allow_fixture:
        raise ValueError("The bundled fixture dataset is a pipeline example and cannot be submitted. Capture or import a learner dataset with boardProfile virtual-wand-v1, original-lsm9ds1, or rev2-bmi270.")

    sessions = {session["id"]: session for session in data["sessions"]}
    if len(sessions) != len(data["sessions"]):
        raise ValueError("Session IDs must be unique")

    counts: Counter[tuple[str, str]] = Counter()
    session_splits: defaultdict[str, set[str]] = defaultdict(set)
    stroke_ids: set[str] = set()
    for stroke in data["strokes"]:
        if stroke["id"] in stroke_ids:
            raise ValueError(f"Duplicate stroke ID: {stroke['id']}")
        stroke_ids.add(stroke["id"])
        session = sessions.get(stroke["sessionId"])
        if not session:
            raise ValueError(f"Unknown session: {stroke['sessionId']}")
        if session["label"] != stroke["label"]:
            raise ValueError(f"Label mismatch for {stroke['id']}")
        counts[(stroke["label"], session["split"])] += 1
        session_splits[stroke["sessionId"]].add(session["split"])

    leaking = [session_id for session_id, splits in session_splits.items() if len(splits) != 1]
    if leaking:
        raise ValueError(f"Session leakage: {', '.join(leaking)}")
    shortfalls = [f"{label}/{split}: {counts[(label, split)]} < {minimum}" for label in LABELS for split, minimum in EXPECTED.items() if counts[(label, split)] < minimum]
    if shortfalls:
        raise ValueError("Count requirements failed:\n- " + "\n- ".join(shortfalls))

    report = {
        "schemaVersion": data["schemaVersion"],
        "boardProfile": data["boardProfile"],
        "strokeCount": len(data["strokes"]),
        "sessionCount": len(data["sessions"]),
        "counts": {label: {split: counts[(label, split)] for split in EXPECTED} for label in LABELS},
        "sessionLeakageCount": 0,
        "valid": True,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an EmbeddedVille Magic Wand dataset")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-fixture", action="store_true", help="Permit the bundled non-submittable fixture for course-maintainer checks")
    args = parser.parse_args()
    report = validate_dataset(args.dataset, allow_fixture=args.allow_fixture)
    encoded = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
