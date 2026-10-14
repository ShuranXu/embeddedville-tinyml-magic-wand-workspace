from __future__ import annotations
import json
from pathlib import Path
import re

ROOT = Path(__file__).parents[1]
DENIED = {"evaluator", "evidence", "solutions", "staff", "source_map.md", "visual_source_audit.md", "pipeline_reference.py", "train_magic_wand.executed.ipynb"}
PATTERNS = [re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"), re.compile(rb"(?i)(?:api[_-]?key|token|password)\s*[:=]\s*['\"][^'\"]{12,}")]
cfg = json.loads((ROOT / ".embeddedville-workspace.json").read_text())
assert cfg["repository"] == "ShuranXu/embeddedville-tinyml-magic-wand-workspace"
assert cfg["starterVersion"] == "1.2.0"
for path in ROOT.rglob("*"):
    if ".git" in path.parts or not path.is_file(): continue
    names = {part.lower() for part in path.relative_to(ROOT).parts}
    assert not names.intersection(DENIED), f"denied public path: {path}"
    data = path.read_bytes()
    for pattern in PATTERNS: assert not pattern.search(data), f"credential-like content: {path}"
print("Public TinyML workspace boundary validated.")
