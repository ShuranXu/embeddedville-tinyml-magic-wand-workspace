from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from validate_dataset import LABELS, validate_dataset

SEED = 20261001
IMAGE_SIZE = 32
ALLOWED_OPS = {"CONV_2D", "MEAN", "FULLY_CONNECTED", "SOFTMAX"}


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()


def rasterize(points: list[dict[str, float]]) -> np.ndarray:
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    count = len(points)
    for index, (start, end) in enumerate(zip(points, points[1:])):
        x0 = int(round((start["x"] / 0.6 + 1) * IMAGE_SIZE / 2))
        y0 = int(round((-start["y"] / 0.6 + 1) * IMAGE_SIZE / 2))
        x1 = int(round((end["x"] / 0.6 + 1) * IMAGE_SIZE / 2))
        y1 = int(round((-end["y"] / 0.6 + 1) * IMAGE_SIZE / 2))
        t = index / max(1, count)
        color = (int((1 - 2 * t) * 255), int(2 * t * 255), 0) if t < 0.5 else (0, int((2 - 2 * t) * 255), int((2 * t - 1) * 255))
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for step in range(steps + 1):
            x = int(round(x0 + (x1 - x0) * step / steps))
            y = int(round(y0 + (y1 - y0) * step / steps))
            if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
                image[y, x] = np.asarray(color, dtype=np.uint8)
    return image


def load_dataset(path: Path, *, allow_fixture: bool = False) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    validate_dataset(path, allow_fixture=allow_fixture)
    data = json.loads(path.read_text(encoding="utf-8"))
    sessions = {session["id"]: session for session in data["sessions"]}
    images: dict[str, list[np.ndarray]] = {split: [] for split in ("train", "validation", "test")}
    labels: dict[str, list[int]] = {split: [] for split in images}
    for stroke in data["strokes"]:
        split = sessions[stroke["sessionId"]]["split"]
        images[split].append(rasterize(stroke["points"]))
        labels[split].append(LABELS.index(stroke["label"]))
    return ({split: np.asarray(values, dtype=np.float32) for split, values in images.items()}, {split: np.asarray(values, dtype=np.int64) for split, values in labels.items()})


def augment(images: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    augmented = [images]
    for shift_x, shift_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = np.roll(images, shift=(shift_y, shift_x), axis=(1, 2))
        if shift_y > 0: shifted[:, :shift_y] = 0
        if shift_y < 0: shifted[:, shift_y:] = 0
        if shift_x > 0: shifted[:, :, :shift_x] = 0
        if shift_x < 0: shifted[:, :, shift_x:] = 0
        augmented.append(shifted)
    return np.concatenate(augmented), np.tile(labels, len(augmented))


def make_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="stroke_image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    for filters in (16, 32, 64):
        x = tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu")(x)
        x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(len(LABELS), activation="softmax", name="gesture_scores")(x)
    return tf.keras.Model(inputs, outputs, name="magic_wand_cnn")


def representative(images: np.ndarray):
    def generator():
        for image in images[: min(160, len(images))]:
            yield [image[np.newaxis].astype(np.float32)]
    return generator


def export_int8(model: tf.keras.Model, train_images: np.ndarray, path: Path, saved_model: Path) -> bytes:
    if saved_model.exists(): shutil.rmtree(saved_model)
    model.export(saved_model)
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative(train_images)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()
    path.write_bytes(model_bytes)
    return model_bytes


def tflite_predictions(model_bytes: bytes, images: np.ndarray) -> tuple[np.ndarray, dict, set[str]]:
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    predictions = []
    input_scale, input_zero = input_info["quantization"]
    output_scale, output_zero = output_info["quantization"]
    for image in images:
        quantized = np.clip(np.round(image / input_scale + input_zero), -128, 127).astype(np.int8)[np.newaxis]
        interpreter.set_tensor(input_info["index"], quantized)
        interpreter.invoke()
        output = interpreter.get_tensor(output_info["index"])[0].astype(np.float32)
        predictions.append((output - output_zero) * output_scale)
    # XNNPACK exposes a synthetic DELEGATE entry on desktop. It is not stored in
    # the FlatBuffer and therefore is not an operator the firmware must register.
    ops = {
        entry["op_name"]
        for entry in interpreter._get_ops_details()  # Audited build contract.
        if entry["op_name"] != "DELEGATE"
    }
    metadata = {
        "inputShape": input_info["shape"].tolist(), "inputDtype": str(input_info["dtype"].__name__),
        "outputShape": output_info["shape"].tolist(), "outputDtype": str(output_info["dtype"].__name__),
        "inputQuantization": [input_scale, int(input_zero)], "outputQuantization": [output_scale, int(output_zero)],
    }
    return np.asarray(predictions), metadata, ops


def write_c_array(model_bytes: bytes, output_dir: Path) -> None:
    rows = []
    for offset in range(0, len(model_bytes), 12):
        rows.append("  " + ", ".join(f"0x{byte:02x}" for byte in model_bytes[offset:offset + 12]) + ",")
    header = "#ifndef MAGIC_WAND_MODEL_DATA_H\n#define MAGIC_WAND_MODEL_DATA_H\n\n#include <cstdint>\nextern const unsigned char g_magic_wand_model_data[];\nextern const int g_magic_wand_model_data_len;\n\n#endif\n"
    source = '#include "magic_wand_model_data.h"\n\nalignas(16) const unsigned char g_magic_wand_model_data[] = {\n' + "\n".join(rows) + f"\n}};\nconst int g_magic_wand_model_data_len = {len(model_bytes)};\n"
    (output_dir / "magic_wand_model_data.h").write_text(header, encoding="utf-8")
    (output_dir / "magic_wand_model_data.cpp").write_text(source, encoding="utf-8")
    labels = "#ifndef MAGIC_WAND_MODEL_LABELS_H\n#define MAGIC_WAND_MODEL_LABELS_H\n\nconstexpr int label_count = 4;\nconstexpr const char* labels[label_count] = {\"wing\", \"ring\", \"slope\", \"unknown\"};\n\n#endif\n"
    (output_dir / "model_labels.h").write_text(labels, encoding="utf-8")


def plot_history(history: tf.keras.callbacks.History, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train"); axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set(title="Gesture classifier accuracy", xlabel="Epoch", ylabel="Accuracy"); axes[0].legend(); axes[0].grid(alpha=0.2)
    axes[1].plot(history.history["loss"], label="train"); axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set(title="Gesture classifier loss", xlabel="Epoch", ylabel="Cross-entropy"); axes[1].legend(); axes[1].grid(alpha=0.2)
    figure.tight_layout(); figure.savefig(output, dpi=150); plt.close(figure)


def write_evaluation_artifacts(output_dir: Path, model_bytes: bytes, metadata: dict, ops: set[str], scores: np.ndarray, labels: np.ndarray, host_predictions: np.ndarray) -> dict:
    predicted = np.argmax(scores, axis=1)
    matrix = confusion_matrix(labels, predicted, labels=list(range(len(LABELS))))
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    macro_accuracy = float(np.mean(predicted == labels))
    host_export_agreement = float(np.mean(host_predictions == predicted))
    metrics = {"seed": SEED, "labels": list(LABELS), "macroAccuracy": macro_accuracy, "perClassRecall": {label: float(recall[index]) for index, label in enumerate(LABELS)}, "confusionMatrix": matrix.tolist(), "tensorMetadata": metadata, "operators": sorted(ops), "modelBytes": len(model_bytes), "modelSha256": hashlib.sha256(model_bytes).hexdigest(), "parity": {"evaluatedExamples": int(len(predicted)), "hostExportAgreement": host_export_agreement}}
    if metadata["inputShape"] != [1, 32, 32, 3] or metadata["outputShape"] != [1, 4] or metadata["inputDtype"] != "int8" or metadata["outputDtype"] != "int8": raise RuntimeError(f"Tensor contract failed: {metadata}")
    if ops - ALLOWED_OPS: raise RuntimeError(f"Unsupported operators: {sorted(ops - ALLOWED_OPS)}")
    if len(model_bytes) > 64 * 1024: raise RuntimeError(f"Model is too large: {len(model_bytes)} bytes")
    if host_export_agreement < 0.99: raise RuntimeError(f"Host/export parity failed: {host_export_agreement:.1%}")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output_dir / "parity-report.json").write_text(json.dumps({"schemaVersion": 1, "modelSha256": metrics["modelSha256"], **metrics["parity"]}, indent=2) + "\n", encoding="utf-8")
    (output_dir / "model_card.md").write_text(f"# Magic Wand model card\n\n- Classes: {', '.join(LABELS)}\n- Input: `int8[1,32,32,3]`\n- Output: `int8[1,4]`\n- Test accuracy: {macro_accuracy:.1%}\n- Model bytes: {len(model_bytes)}\n- SHA-256: `{metrics['modelSha256']}`\n- Operators: {', '.join(sorted(ops))}\n\nThis card describes the dataset and model generated in the same run. The bundled fixture is not learner submission evidence.\n", encoding="utf-8")
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(4), yticks=range(4), xticklabels=LABELS, yticklabels=LABELS, xlabel="Predicted", ylabel="Actual", title="Untouched test-session confusion matrix")
    for row in range(4):
        for column in range(4): axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis); figure.tight_layout(); figure.savefig(output_dir / "confusion_matrix.png", dpi=150); plt.close(figure)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export the EmbeddedVille Magic Wand classifier")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--firmware-dir", type=Path)
    parser.add_argument("--allow-fixture", action="store_true", help="Maintainer-only: train the bundled non-submittable fixture")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything()
    images, label_ids = load_dataset(args.dataset, allow_fixture=args.allow_fixture)
    train_images, train_labels = augment(images["train"], label_ids["train"])
    model = make_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, mode="max", restore_best_weights=True)]
    history = model.fit(train_images, train_labels, validation_data=(images["validation"], label_ids["validation"]), epochs=args.epochs, batch_size=32, callbacks=callbacks, verbose=2)
    model.save(args.output_dir / "magic_wand.keras")
    plot_history(history, args.output_dir / "learning_curves.png")
    model_bytes = export_int8(model, images["train"], args.output_dir / "magic_wand_int8.tflite", args.output_dir / "saved_model")
    scores, metadata, ops = tflite_predictions(model_bytes, images["test"])
    host_predicted = np.argmax(model.predict(images["test"], verbose=0), axis=1)
    metrics = write_evaluation_artifacts(args.output_dir, model_bytes, metadata, ops, scores, label_ids["test"], host_predicted)
    write_c_array(model_bytes, args.output_dir)
    if args.firmware_dir:
        for name in ("magic_wand_model_data.h", "magic_wand_model_data.cpp", "model_labels.h"):
            shutil.copy2(args.output_dir / name, args.firmware_dir / name)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
