from pathlib import Path

import nbformat as nbf

root = Path(__file__).parent
notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "colab": {"name": "TinyML Magic Wand Training", "provenance": []},
}
notebook["cells"] = [
    nbf.v4.new_markdown_cell("# TinyML Magic Wand: train and export\n\nThis notebook validates a session-grouped four-class dataset, trains the 32×32×3 classifier, converts it to full int8, and generates the exact firmware model files."),
    nbf.v4.new_markdown_cell("## 1. Environment and reproducibility\n\nUse Python 3.11 and the pinned packages in `requirements.txt`. The fixed seed makes the fixture run repeatable; real learner metrics still depend on captured data."),
    nbf.v4.new_code_cell("from pathlib import Path\nimport json\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nTRAINING = Path.cwd() if Path('requirements.txt').exists() else Path('training')\nif not (TRAINING / 'requirements.txt').exists():\n    raise RuntimeError('Run this notebook from the starter root or its training directory.')\nimport sys\nsys.path.insert(0, str(TRAINING.resolve()))\nfrom validate_dataset import validate_dataset\nfrom train_magic_wand import LABELS, SEED, load_dataset, seed_everything, make_model, augment, export_int8, tflite_predictions, write_c_array, write_evaluation_artifacts\n\nDATASET = TRAINING / 'learner_dataset.json'\nOUTPUT = TRAINING / 'artifacts'\nOUTPUT.mkdir(exist_ok=True)\nif not DATASET.exists():\n    raise FileNotFoundError('Save the Lab 2 recorder export as training/learner_dataset.json before training.')\nseed_everything()\nprint({'tensorflow_seed': SEED, 'labels': LABELS, 'dataset': str(DATASET), 'output': str(OUTPUT)})"),
    nbf.v4.new_markdown_cell("## 2. Validate the dataset contract\n\nValidation checks schema, unique IDs, balanced counts, point bounds, and session-grouped splits before any training begins."),
    nbf.v4.new_code_cell("report = validate_dataset(DATASET)\n(OUTPUT / 'dataset-validator-report.json').write_text(json.dumps(report, indent=2) + '\\n')\nreport"),
    nbf.v4.new_markdown_cell("## 3. Rasterize and inspect examples\n\nThe Python rasterizer preserves stroke order as red → green → blue, matching the firmware representation."),
    nbf.v4.new_code_cell("images, label_ids = load_dataset(DATASET)\nfig, axes = plt.subplots(1, 4, figsize=(12, 3))\nfor class_index, (label, axis) in enumerate(zip(LABELS, axes)):\n    image = images['train'][label_ids['train'] == class_index][0].astype('uint8')\n    axis.imshow(image); axis.set_title(label); axis.axis('off')\nfig.suptitle('Session-grouped training examples'); fig.tight_layout()"),
    nbf.v4.new_markdown_cell("## 4. Train and select the best checkpoint\n\nOnly training examples are augmented. Validation and test sessions remain untouched."),
    nbf.v4.new_code_cell("import tensorflow as tf\ntrain_images, train_labels = augment(images['train'], label_ids['train'])\nmodel = make_model()\nmodel.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])\nhistory = model.fit(train_images, train_labels, validation_data=(images['validation'], label_ids['validation']), epochs=30, batch_size=32, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max', restore_best_weights=True)], verbose=2)"),
    nbf.v4.new_code_cell("fig, axes = plt.subplots(1, 2, figsize=(10, 4))\naxes[0].plot(history.history['accuracy'], label='train'); axes[0].plot(history.history['val_accuracy'], label='validation'); axes[0].set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy'); axes[0].legend(); axes[0].grid(alpha=.2)\naxes[1].plot(history.history['loss'], label='train'); axes[1].plot(history.history['val_loss'], label='validation'); axes[1].set(title='Loss', xlabel='Epoch', ylabel='Cross-entropy'); axes[1].legend(); axes[1].grid(alpha=.2)\nfig.tight_layout(); fig.savefig(OUTPUT / 'learning_curves.png', dpi=150)"),
    nbf.v4.new_markdown_cell("## 5. Convert and audit the full-int8 model\n\nThe exported model must have `int8[1,32,32,3]` input, four int8 outputs, no unsupported operators, and a size no greater than 64 KB."),
    nbf.v4.new_code_cell("model_bytes = export_int8(model, images['train'], OUTPUT / 'magic_wand_int8.tflite', OUTPUT / 'saved_model')\nscores, metadata, operators = tflite_predictions(model_bytes, images['test'])\nmetadata, sorted(operators), len(model_bytes)"),
    nbf.v4.new_markdown_cell("## 6. Evaluate the untouched test session\n\nReport overall accuracy, per-class recall, and the complete confusion matrix. Do not tune against this partition."),
    nbf.v4.new_code_cell("from sklearn.metrics import classification_report\npredicted = np.argmax(scores, axis=1)\nprint(classification_report(label_ids['test'], predicted, target_names=LABELS, digits=3))\nhost_predicted = np.argmax(model.predict(images['test'], verbose=0), axis=1)\nmetrics = write_evaluation_artifacts(OUTPUT, model_bytes, metadata, operators, scores, label_ids['test'], host_predicted)\nmetrics"),
    nbf.v4.new_markdown_cell("## 7. Generate firmware artifacts\n\nThe C array and labels are generated together so the output order cannot drift."),
    nbf.v4.new_code_cell("write_c_array(model_bytes, OUTPUT)\nsummary = {'modelBytes': len(model_bytes), 'sha256': metrics['modelSha256'], 'operators': sorted(operators), 'labels': list(LABELS), 'generated': ['learning_curves.png', 'confusion_matrix.png', 'metrics.json', 'parity-report.json', 'model_card.md', 'magic_wand_int8.tflite', 'magic_wand_model_data.cpp', 'magic_wand_model_data.h', 'model_labels.h']}\n(OUTPUT / 'notebook_summary.json').write_text(json.dumps(summary, indent=2) + '\\n')\nsummary"),
    nbf.v4.new_markdown_cell("## 8. Deployment handoff\n\nThe notebook has written the complete Lab 3 evidence set to `training/artifacts/`. Copy the generated C files and labels into the firmware only after checking the tensor, operator, size, accuracy, recall, and parity gates. Then load the same `.tflite` and `metrics.json` in the Virtual Wand, run the untouched test split, and save the exported report as `training/artifacts/virtual-wand-report.json`."),
]
for index, cell in enumerate(notebook["cells"]):
    cell["id"] = f"tinyml-{index:02d}"
(root / "train_magic_wand.ipynb").write_text(nbf.writes(notebook), encoding="utf-8")
print(root / "train_magic_wand.ipynb")
