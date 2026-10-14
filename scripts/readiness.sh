#!/usr/bin/env bash
set -euo pipefail
python3 -m py_compile ./lab training/*.py
python3 training/validate_dataset.py training/fixture_dataset.json --allow-fixture
node --test simulator/simulator-core.test.mjs
python3 tests/validate_workspace.py
printf 'Workspace ready: TinyML starter 1.2.0\n'
