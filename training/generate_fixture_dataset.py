from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

LABELS = ("wing", "ring", "slope", "unknown")
SPLITS = (("train", 40), ("validation", 10), ("test", 10))


def base_point(label: str, t: float, variant: int) -> tuple[float, float]:
    if label == "ring":
        return 0.62 * math.cos(2 * math.pi * t), 0.62 * math.sin(2 * math.pi * t)
    if label == "slope":
        return -0.7 + 1.4 * t, -0.55 + 1.1 * t
    if label == "wing":
        if t < 0.5:
            u = t * 2
            return -0.72 + 0.72 * u, -0.45 + 0.95 * u
        u = (t - 0.5) * 2
        return 0.0 + 0.72 * u, 0.50 - 0.95 * u
    mode = variant % 3
    if mode == 0:
        return 0.65 * math.sin(4 * math.pi * t), -0.65 + 1.3 * t
    if mode == 1:
        return -0.65 + 1.3 * t, 0.45 * math.sin(3 * math.pi * t)
    return 0.55 * math.cos(3 * math.pi * t), 0.55 * math.sin(5 * math.pi * t)


def make_stroke(label: str, index: int, seed: int) -> list[dict[str, float]]:
    rng = random.Random(seed)
    count = 64 + index % 18
    scale = rng.uniform(0.83, 1.05)
    rotation = rng.uniform(-0.13, 0.13)
    dx, dy = rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08)
    points = []
    for point_index in range(count):
        t = point_index / (count - 1)
        x, y = base_point(label, t, index)
        rx = scale * (x * math.cos(rotation) - y * math.sin(rotation)) + dx + rng.gauss(0, 0.018)
        ry = scale * (x * math.sin(rotation) + y * math.cos(rotation)) + dy + rng.gauss(0, 0.018)
        points.append({"x": round(max(-1, min(1, rx)), 5), "y": round(max(-1, min(1, ry)), 5), "t": round(t, 5)})
    return points


def main() -> None:
    sessions = []
    strokes = []
    seed = 20261001
    for label_index, label in enumerate(LABELS):
        offset = 0
        for split_index, (split, count) in enumerate(SPLITS):
            session_id = f"fixture-{label}-{split}"
            sessions.append({"id": session_id, "label": label, "split": split})
            for local_index in range(count):
                index = offset + local_index
                strokes.append({"id": f"{label}-{split}-{local_index:03d}", "label": label, "sessionId": session_id, "points": make_stroke(label, index, seed + label_index * 1000 + split_index * 100 + local_index)})
            offset += count
    data = {"schemaVersion": 1, "exportedAt": datetime.now(timezone.utc).isoformat(), "boardProfile": "fixture", "sessions": sessions, "strokes": strokes}
    output = Path(__file__).parent / "fixture_dataset.json"
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(strokes)} strokes to {output}")


if __name__ == "__main__":
    main()
