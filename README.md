# EmbeddedVille TinyML Magic Wand workspace

Public, course-scoped workspace for the TinyML Magic Wand final project. It contains the learner firmware, recorder, browser simulator, dataset schema, training notebook, and training pipeline. Private evaluator scenarios, reference answers, release evidence, authoring notes, and credentials are intentionally absent.

## Start

Open the immutable Codespaces link from the course. The container installs the pinned Python environment and runs the public readiness checks.

```bash
./lab check --activity tinyml-project-book-gestures
python -m http.server 8000
```

Open port 8000, then use `/recorder/` and `/simulator/` as directed by the course. The bundled fixture dataset is for pipeline readiness only and is not accepted as learner project evidence.

Submit the current allowlisted workspace, including uncommitted edits:

```bash
./lab submit --activity tinyml-project-book-gestures
```

The command shows a ten-minute verification URL and code. Sign in and approve only after checking the exact activity, starter version, source state, and snapshot digest.

Saving a file is not submission. A stopped Codespace normally retains saved files, while deletion or retention expiry can remove unpushed work. Stop compute when finished. GitHub account, Codespaces availability, quota, billing, and retention follow your GitHub plan and organization policy.

Only allowlisted paths enter the snapshot. `.git`, environment files, credentials, private keys, caches, generated bulk output, symlinks, and evaluator content are rejected.
