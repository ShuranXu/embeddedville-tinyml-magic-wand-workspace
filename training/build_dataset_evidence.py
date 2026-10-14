from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from validate_dataset import LABELS, EXPECTED, validate_dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a learner dataset and generate its manifest and sample gallery")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-fixture", action="store_true", help="Maintainer-only: generate evidence for the bundled demo fixture")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = validate_dataset(args.dataset, allow_fixture=args.allow_fixture)
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    sessions = {session["id"]: session for session in data["sessions"]}
    manifest = {
        "schemaVersion": 1,
        "datasetFile": args.dataset.name,
        "datasetSha256": sha256(args.dataset),
        "boardProfile": data["boardProfile"],
        "strokeCount": report["strokeCount"],
        "sessionCount": report["sessionCount"],
        "counts": report["counts"],
        "sessionLeakageCount": report["sessionLeakageCount"],
        "submittable": data["boardProfile"] != "fixture",
    }
    (args.output_dir / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "dataset-validator-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(len(LABELS), len(EXPECTED), figsize=(10, 10), squeeze=False)
    for row, label in enumerate(LABELS):
        for column, split in enumerate(EXPECTED):
            axis = axes[row][column]
            sample = next(stroke for stroke in data["strokes"] if stroke["label"] == label and sessions[stroke["sessionId"]]["split"] == split)
            x = [point["x"] for point in sample["points"]]
            y = [point["y"] for point in sample["points"]]
            axis.plot(x, y, color="#05869b", linewidth=2)
            axis.scatter([x[0], x[-1]], [y[0], y[-1]], c=["#1b8f62", "#c77700"], s=30)
            axis.set_title(f"{label} · {split}")
            axis.set_xlim(-0.7, 0.7); axis.set_ylim(-0.7, 0.7); axis.set_aspect("equal"); axis.grid(alpha=.18)
    figure.suptitle("Magic Wand dataset · one reviewed example per class and split")
    figure.tight_layout()
    figure.savefig(args.output_dir / "sample-gallery.png", dpi=150)
    plt.close(figure)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
