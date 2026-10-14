import { IMAGE_SIZE, LABELS, classifyScores, normalizeStroke, quantizeRaster, rasterizeStroke, summarizeTrials, validateDataset } from "./simulator-core.js";

const COURSE_MODEL = "magic_wand_int8.tflite";
const COURSE_METRICS = "metrics.json";
const WASM_PATH = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/";
const CAPTURE_STORAGE_KEY = "embeddedville:tinyml-magic-wand:capture:v1";
const $ = (id) => document.getElementById(id);
const labProfiles = {
  "tinyml-lab-dataset": { title: "Lab 2 · Dataset Capture and Replay Sandbox", body: "Capture session-grouped virtual strokes, export schema-v1 data, inspect examples by split, and replay the untouched test set." },
  "tinyml-lab-model": { title: "Lab 3 · Model Deployment Sandbox", body: "Load your int8 FlatBuffer and matching metrics, audit its contract, run real TensorFlow Lite inference, and export evidence bound to its SHA-256." },
  "tinyml-project-book-gestures": { title: "Final Project · Virtual Wand Capstone", body: "Run ten interactive trials for each named gesture and twenty unknown trials with one fixed model hash and threshold policy." },
};
const state = {
  model: null,
  modelName: "",
  modelBytes: null,
  modelHash: "",
  metrics: null,
  dataset: null,
  points: [],
  raster: new Uint8Array(IMAGE_SIZE * IMAGE_SIZE * 3),
  rawDrawPoints: [],
  trials: [],
  captured: {
    schemaVersion: 1,
    exportedAt: null,
    boardProfile: "virtual-wand-v1",
    sessions: [],
    strokes: [],
  },
  drawing: false,
  currentSource: "drawn gesture",
};

const splitTargets = { train: 40, validation: 10, test: 10 };

function emptyCapturedDataset() {
  return { schemaVersion: 1, exportedAt: null, boardProfile: "virtual-wand-v1", sessions: [], strokes: [] };
}

function isRecoverableCapture(value) {
  return value && value.schemaVersion === 1 && value.boardProfile === "virtual-wand-v1" && Array.isArray(value.sessions) && Array.isArray(value.strokes);
}

function saveCaptureDraft(message = "Draft autosaved in this browser.") {
  try {
    localStorage.setItem(CAPTURE_STORAGE_KEY, JSON.stringify({ storageVersion: 1, savedAt: new Date().toISOString(), dataset: state.captured }));
    $("capture-storage-status").textContent = `${message} ${state.captured.strokes.length} examples are recoverable after reload.`;
  } catch {
    $("capture-storage-status").textContent = "Browser autosave is unavailable. Export the dataset now to avoid losing captured work.";
  }
}

function restoreCaptureDraft() {
  try {
    const stored = JSON.parse(localStorage.getItem(CAPTURE_STORAGE_KEY) ?? "null");
    if (stored?.storageVersion === 1 && isRecoverableCapture(stored.dataset)) {
      state.captured = stored.dataset;
      return `Restored ${state.captured.strokes.length} autosaved examples from this browser.`;
    }
  } catch {
    localStorage.removeItem(CAPTURE_STORAGE_KEY);
  }
  return "Draft recovery is ready. Captures autosave in this browser until you export or clear them.";
}

const defaultMetrics = {
  labels: LABELS,
  operators: ["CONV_2D", "FULLY_CONNECTED", "MEAN", "SOFTMAX"],
  tensorMetadata: { inputQuantization: [1, -128], outputQuantization: [0.00390625, -128] },
};

function configureLabContext() {
  const labId = new URLSearchParams(window.location.search).get("lab");
  const profile = labProfiles[labId];
  if (!profile) return;
  $("lab-context-title").textContent = profile.title;
  $("lab-context-body").textContent = profile.body;
  $("lab-context").hidden = false;
  document.title = `${profile.title} · EmbeddedVille`;
}

function setStatus(message, error = false) {
  $("status").textContent = message;
  $("status").style.color = error ? "#9b3941" : "#506b78";
}

async function sha256(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

function thresholds() {
  return { confidence: Number($("confidence").value), margin: Number($("margin").value) };
}

function drawStroke() {
  const canvas = $("stroke");
  const context = canvas.getContext("2d");
  context.fillStyle = "#061827";
  context.fillRect(0, 0, canvas.width, canvas.height);
  if (state.points.length < 2) return;
  context.lineWidth = 7;
  context.lineCap = "round";
  context.lineJoin = "round";
  for (let index = 0; index < state.points.length - 1; index += 1) {
    const start = state.points[index];
    const end = state.points[index + 1];
    context.strokeStyle = `hsl(${240 * index / Math.max(1, state.points.length - 1)},88%,60%)`;
    context.beginPath();
    context.moveTo((start.x / 0.6 + 1) * canvas.width / 2, (-start.y / 0.6 + 1) * canvas.height / 2);
    context.lineTo((end.x / 0.6 + 1) * canvas.width / 2, (-end.y / 0.6 + 1) * canvas.height / 2);
    context.stroke();
  }
}

function drawRaster() {
  const canvas = $("raster");
  const context = canvas.getContext("2d");
  const image = context.createImageData(IMAGE_SIZE, IMAGE_SIZE);
  for (let pixel = 0; pixel < IMAGE_SIZE * IMAGE_SIZE; pixel += 1) {
    image.data[pixel * 4] = state.raster[pixel * 3];
    image.data[pixel * 4 + 1] = state.raster[pixel * 3 + 1];
    image.data[pixel * 4 + 2] = state.raster[pixel * 3 + 2];
    image.data[pixel * 4 + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  const quantized = quantizeRaster(state.raster, (state.metrics ?? defaultMetrics).tensorMetadata.inputQuantization);
  $("point-count").textContent = `${state.points.length} points`;
  $("pixel-range").textContent = `${Math.min(...state.raster)}…${Math.max(...state.raster)}`;
  $("tensor-range").textContent = `${Math.min(...quantized)}…${Math.max(...quantized)}`;
  $("run-one").disabled = !state.model || state.points.length < 2;
}

function setPoints(points) {
  state.points = points.map((point, index) => ({ x: Number(point.x), y: Number(point.y), t: Number.isFinite(point.t) ? Number(point.t) : index / Math.max(1, points.length - 1) }));
  state.raster = rasterizeStroke(state.points);
  drawStroke();
  drawRaster();
}

function renderScores(result) {
  const scoreRoot = $("scores");
  scoreRoot.replaceChildren();
  const labels = state.metrics?.labels ?? LABELS;
  result.scores.forEach((score, index) => {
    const row = document.createElement("div");
    row.className = "score";
    const label = document.createElement("strong"); label.textContent = labels[index];
    const track = document.createElement("i"); const bar = document.createElement("b"); bar.style.width = `${Math.max(0, Math.min(100, score * 100))}%`; track.append(bar);
    const output = document.createElement("output"); output.textContent = score.toFixed(3);
    row.append(label, track, output); scoreRoot.append(row);
  });
  const decision = $("decision");
  decision.classList.toggle("rejected", !result.accepted);
  decision.replaceChildren();
  const caption = document.createElement("span"); caption.textContent = result.accepted ? "Accepted prediction" : "Rejected to unknown";
  const prediction = document.createElement("strong"); prediction.textContent = result.predictedLabel;
  const detail = document.createElement("small"); detail.textContent = `Top ${result.topLabel} · confidence ${result.confidence.toFixed(3)} · margin ${result.margin.toFixed(3)}`;
  decision.append(caption, prediction, detail);
}

function outputTensor(value) {
  if (Array.isArray(value)) return value[0];
  if (value && typeof value.data === "function") return value;
  return Object.values(value ?? {})[0];
}

async function infer(points, source, actualLabel, record = true) {
  if (!state.model) throw new Error("Load a TFLite model first");
  const raster = rasterizeStroke(points);
  const metadata = state.metrics?.tensorMetadata ?? defaultMetrics.tensorMetadata;
  const input = tf.tensor(quantizeRaster(raster, metadata.inputQuantization), [1, IMAGE_SIZE, IMAGE_SIZE, 3], "int32");
  let prediction;
  try {
    prediction = state.model.predict(input);
    const tensor = outputTensor(prediction);
    if (!tensor) throw new Error("The model did not return an output tensor");
    const raw = await tensor.data();
    const result = classifyScores(raw, state.metrics?.labels ?? LABELS, metadata.outputQuantization, thresholds());
    if (record) {
      state.trials.push({ number: state.trials.length + 1, source, actualLabel, predictedLabel: result.predictedLabel, topLabel: result.topLabel, confidence: result.confidence, margin: result.margin, scores: result.scores });
      renderTrials();
    }
    return result;
  } finally {
    input.dispose();
    if (Array.isArray(prediction)) prediction.forEach((tensor) => tensor.dispose());
    else if (prediction && typeof prediction.dispose === "function") prediction.dispose();
    else Object.values(prediction ?? {}).forEach((tensor) => tensor?.dispose?.());
  }
}

async function runCurrent() {
  if (state.points.length < 2) return;
  try {
    $("run-one").disabled = true;
    const result = await infer(state.points, state.currentSource, $("actual-label").value);
    renderScores(result);
    setStatus(`Inference complete: ${result.predictedLabel}. The trial was added to the simulation report.`);
  } catch (error) {
    setStatus(`Inference failed: ${error.message}`, true);
  } finally {
    $("run-one").disabled = !state.model || state.points.length < 2;
  }
}

function renderContract() {
  const input = state.model?.inputs?.[0];
  const output = state.model?.outputs?.[0];
  const metadata = state.metrics?.tensorMetadata ?? defaultMetrics.tensorMetadata;
  const inputType = metadata.inputDtype === "int8" && input?.dtype === "int32" ? "int8 · browser int32 proxy" : input?.dtype;
  const outputType = metadata.outputDtype === "int8" && output?.dtype === "int32" ? "int8 · browser int32 proxy" : output?.dtype;
  $("model-name").textContent = state.modelName || "No model loaded";
  $("model-hash").textContent = state.modelHash ? `SHA-256 ${state.modelHash}` : "SHA-256 pending";
  $("input-contract").textContent = input ? `${inputType} [${input.shape.join(" × ")}]` : "—";
  $("output-contract").textContent = output ? `${outputType} [${output.shape.join(" × ")}]` : "—";
  $("input-quantization").textContent = `scale ${metadata.inputQuantization[0]} · zero ${metadata.inputQuantization[1]}`;
  $("output-quantization").textContent = `scale ${metadata.outputQuantization[0]} · zero ${metadata.outputQuantization[1]}`;
  $("operators").textContent = state.metrics?.operators?.join(" · ") ?? "Load metrics to audit";
}

async function loadModel(bytes, name, metrics = undefined) {
  if (!globalThis.tflite || !globalThis.tf) throw new Error("The TensorFlow Lite browser runtime did not load. Check network access and reload.");
  await tf.setBackend("cpu");
  await tf.ready();
  setStatus(`Loading ${name} in the WebAssembly interpreter…`);
  const model = await tflite.loadTFLiteModel(bytes, { numThreads: 1 });
  const input = model.inputs[0];
  const output = model.outputs[0];
  if (JSON.stringify(input.shape) !== JSON.stringify([1, 32, 32, 3]) || input.dtype !== "int32") throw new Error(`Expected int8 input [1,32,32,3]; browser contract is ${input.dtype} [${input.shape}]`);
  if (JSON.stringify(output.shape) !== JSON.stringify([1, 4]) || output.dtype !== "int32") throw new Error(`Expected int8 output [1,4]; browser contract is ${output.dtype} [${output.shape}]`);
  if (bytes.byteLength > 64 * 1024) throw new Error(`Model is ${bytes.byteLength} bytes; the course limit is 65536 bytes`);
  if (metrics !== undefined) state.metrics = metrics;
  state.model = model;
  state.modelBytes = bytes;
  state.modelName = name;
  state.modelHash = await sha256(bytes);
  state.trials = [];
  renderContract(); renderTrials(); drawRaster();
  setStatus(`${name} is ready. Draw a gesture, replay a dataset example, or run the untouched test split.`);
}

async function restoreCourseModel() {
  try {
    tflite.setWasmPath(WASM_PATH);
    const [modelResponse, metricsResponse] = await Promise.all([fetch(COURSE_MODEL), fetch(COURSE_METRICS)]);
    if (!modelResponse.ok || !metricsResponse.ok) throw new Error("The bundled course artifacts could not be loaded");
    const metrics = await metricsResponse.json();
    await loadModel(await modelResponse.arrayBuffer(), "course fixture · magic_wand_int8.tflite", metrics);
  } catch (error) {
    setStatus(`Course model unavailable: ${error.message}`, true);
  }
}

function populateDataset() {
  const select = $("stroke-select");
  select.replaceChildren();
  for (const [index, stroke] of state.dataset.strokes.entries()) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${stroke.label} · ${stroke.id} · ${stroke.sessionId}`;
    select.append(option);
  }
  select.disabled = false;
  $("run-dataset").disabled = !state.model;
  $("dataset-summary").textContent = `${state.dataset.strokes.length} strokes across ${state.dataset.sessions.length} sessions. Select one to replay or run the test split.`;
  replaySelected();
}

function replaySelected() {
  if (!state.dataset) return;
  const stroke = state.dataset.strokes[Number($("stroke-select").value)];
  if (!stroke) return;
  $("actual-label").value = stroke.label;
  state.currentSource = `dataset replay · ${stroke.id}`;
  setPoints(stroke.points);
  setStatus(`Replaying ${stroke.id} from ${stroke.sessionId}. Run the model when the raster looks correct.`);
}

async function runDataset() {
  if (!state.dataset || !state.model) return;
  const sessions = new Map(state.dataset.sessions.map((session) => [session.id, session]));
  const testStrokes = state.dataset.strokes.filter((stroke) => sessions.get(stroke.sessionId)?.split === "test");
  if (!testStrokes.length) { setStatus("The imported dataset has no strokes assigned to the test split.", true); return; }
  $("run-dataset").disabled = true;
  state.trials = [];
  try {
    for (let index = 0; index < testStrokes.length; index += 1) {
      const stroke = testStrokes[index];
      const result = await infer(stroke.points, `test · ${stroke.id}`, stroke.label, false);
      state.trials.push({ number: index + 1, source: `test · ${stroke.id}`, actualLabel: stroke.label, predictedLabel: result.predictedLabel, topLabel: result.topLabel, confidence: result.confidence, margin: result.margin, scores: result.scores });
      if (index % 5 === 0) setStatus(`Running untouched test split… ${index + 1}/${testStrokes.length}`);
    }
    renderTrials();
    const summary = summarizeTrials(state.trials, state.metrics?.labels ?? LABELS);
    setStatus(`Test split complete: ${(summary.accuracy * 100).toFixed(1)}% accuracy and ${(summary.unknownFalseAcceptance * 100).toFixed(1)}% unknown false acceptance.`);
  } catch (error) {
    setStatus(`Dataset inference failed: ${error.message}`, true);
  } finally {
    $("run-dataset").disabled = false;
  }
}

function renderTrials() {
  const body = $("trials");
  body.replaceChildren();
  if (!state.trials.length) {
    const row = document.createElement("tr"); const cell = document.createElement("td"); cell.colSpan = 6; cell.textContent = "Run one gesture or a dataset split to populate evidence."; row.append(cell); body.append(row);
    $("trial-summary").textContent = "No trials recorded";
    return;
  }
  for (const trial of state.trials) {
    const row = document.createElement("tr");
    for (const value of [trial.number, trial.source, trial.actualLabel, trial.predictedLabel, trial.confidence.toFixed(3), trial.margin.toFixed(3)]) { const cell = document.createElement("td"); cell.textContent = String(value); row.append(cell); }
    body.append(row);
  }
  const summary = summarizeTrials(state.trials, state.metrics?.labels ?? LABELS);
  $("trial-summary").textContent = `${summary.count} trials · ${(summary.accuracy * 100).toFixed(1)}% accuracy · ${(summary.unknownFalseAcceptance * 100).toFixed(1)}% unknown false acceptance`;
}

function examplePoints(kind) {
  const points = [];
  const count = 64;
  for (let index = 0; index < count; index += 1) {
    const time = index / (count - 1);
    if (kind === "ring") points.push({ x: Math.cos(time * Math.PI * 2), y: Math.sin(time * Math.PI * 2) });
    else if (kind === "slope") points.push({ x: time * 2 - 1, y: time * 1.6 - 0.8 });
    else if (kind === "wing") points.push({ x: time * 2 - 1, y: Math.sin(time * Math.PI * 3) * (0.75 - time * 0.25) });
    else points.push({ x: Math.sin(time * Math.PI * 5) * 0.7, y: Math.cos(time * Math.PI * 3) * 0.5 + time * 0.25 });
  }
  return normalizeStroke(points);
}

function pointerPosition(event) {
  const bounds = $("stroke").getBoundingClientRect();
  return { x: (event.clientX - bounds.left) / bounds.width, y: (event.clientY - bounds.top) / bounds.height };
}

function updateDrawnStroke() {
  setPoints(normalizeStroke(state.rawDrawPoints));
}

function captureSession(label, split) {
  const id = `${label}-${split}-virtual`;
  let session = state.captured.sessions.find((entry) => entry.id === id);
  if (!session) {
    session = { id, label, split };
    state.captured.sessions.push(session);
  }
  return session;
}

function renderCaptureCounts() {
  const root = $("capture-counts");
  root.replaceChildren();
  for (const label of LABELS) {
    const article = document.createElement("article");
    const title = document.createElement("b");
    title.textContent = label;
    const counts = Object.fromEntries(Object.keys(splitTargets).map((split) => [split, state.captured.strokes.filter((stroke) => stroke.label === label && stroke.sessionId === `${label}-${split}-virtual`).length]));
    const detail = document.createElement("small");
    detail.textContent = `train ${counts.train}/40 · validation ${counts.validation}/10 · test ${counts.test}/10`;
    article.append(title, detail);
    root.append(article);
  }
  const total = state.captured.strokes.length;
  $("capture-summary").textContent = `${total} / 240 examples`;
  $("undo-capture").disabled = total === 0;
  $("export-dataset").disabled = total === 0;
  $("review-capture").disabled = total === 0;
  $("remove-capture").disabled = total === 0;
  $("clear-capture").disabled = total === 0;
  const selector = $("capture-select");
  const previous = selector.value;
  selector.replaceChildren();
  if (!total) {
    const option = document.createElement("option"); option.textContent = "No captured examples"; selector.append(option); selector.disabled = true;
  } else {
    state.captured.strokes.forEach((stroke, index) => { const option = document.createElement("option"); option.value = String(index); option.textContent = `${String(index + 1).padStart(3, "0")} · ${stroke.label} · ${stroke.sessionId}`; selector.append(option); });
    selector.disabled = false;
    if ([...selector.options].some((option) => option.value === previous)) selector.value = previous;
  }
}

function captureStroke() {
  if (state.points.length < 20 || state.points.length > 160) {
    setStatus("A captured dataset stroke must contain 20–160 normalized points.", true);
    return;
  }
  const label = $("actual-label").value;
  const split = $("capture-split").value;
  const session = captureSession(label, split);
  const usedIds = new Set(state.captured.strokes.map((stroke) => stroke.id));
  let index = 1;
  while (usedIds.has(`${label}-${split}-${String(index).padStart(3, "0")}`)) index += 1;
  state.captured.strokes.push({
    id: `${label}-${split}-${String(index).padStart(3, "0")}`,
    label,
    sessionId: session.id,
    points: state.points.map((point) => ({ x: Number(point.x.toFixed(6)), y: Number(point.y.toFixed(6)), t: Number(point.t.toFixed(6)) })),
  });
  renderCaptureCounts();
  saveCaptureDraft();
  const count = state.captured.strokes.filter((stroke) => stroke.label === label && stroke.sessionId === session.id).length;
  setStatus(`Captured ${label} for ${split}: ${count}/${splitTargets[split]}. Draw a fresh variation before adding the next example.`);
}

function exportCapturedDataset() {
  const dataset = { ...state.captured, exportedAt: new Date().toISOString() };
  const audit = validateDataset(dataset);
  if (!audit.valid) {
    setStatus(`Dataset is not exportable: ${audit.errors.slice(0, 3).join("; ")}`, true);
    return;
  }
  const url = URL.createObjectURL(new Blob([JSON.stringify(dataset, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `virtual-wand-dataset-${new Date().toISOString().slice(0, 10)}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  setStatus(`Exported ${dataset.strokes.length} virtual examples across ${dataset.sessions.length} sessions.`);
}

function downloadReport() {
  if (!state.modelHash) { setStatus("Load a model before exporting a report.", true); return; }
  const report = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    runtime: "@tensorflow/tfjs-tflite@0.0.1-alpha.10",
    model: { name: state.modelName, sha256: state.modelHash, bytes: state.modelBytes.byteLength, input: state.model.inputs[0], output: state.model.outputs[0], operators: state.metrics?.operators ?? null },
    quantization: state.metrics?.tensorMetadata ?? defaultMetrics.tensorMetadata,
    thresholds: thresholds(),
    summary: summarizeTrials(state.trials, state.metrics?.labels ?? LABELS),
    trials: state.trials,
    evidenceBoundary: "Required virtual-device preprocessing, exported-model inference, thresholds, and interactive trials",
    limitations: ["No physical IMU noise or axis behavior", "No BLE transport timing", "No microcontroller tensor-arena, cycle, or power measurement", "Physical-device testing is an optional extension and is not required for certification"],
  };
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `virtual-wand-report-${new Date().toISOString().slice(0, 10)}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  setStatus("Downloaded the simulation report. Keep it beside the model, metrics, dataset manifest, and build evidence.");
}

$("stroke").addEventListener("pointerdown", (event) => { state.drawing = true; state.currentSource = "drawn gesture"; state.rawDrawPoints = [pointerPosition(event)]; $("stroke").setPointerCapture(event.pointerId); updateDrawnStroke(); });
$("stroke").addEventListener("pointermove", (event) => { if (!state.drawing) return; const point = pointerPosition(event); const last = state.rawDrawPoints.at(-1); if (!last || Math.hypot(point.x - last.x, point.y - last.y) > 0.006) { state.rawDrawPoints.push(point); updateDrawnStroke(); } });
$("stroke").addEventListener("pointerup", () => { state.drawing = false; updateDrawnStroke(); setStatus(`${state.points.length} points normalized. Inspect the raster, choose the actual label, then run the model.`); });
$("clear").onclick = () => { state.currentSource = "drawn gesture"; state.rawDrawPoints = []; setPoints([]); };
document.querySelectorAll("[data-example]").forEach((button) => { button.onclick = () => { const kind = button.dataset.example; state.currentSource = `synthetic ${kind}`; $("actual-label").value = kind; setPoints(examplePoints(kind)); setStatus(`Loaded a synthetic ${kind} stroke. This is a pipeline example, not accuracy evidence.`); }; });
$("run-one").onclick = runCurrent;
$("stroke-select").onchange = replaySelected;
$("run-dataset").onclick = runDataset;
$("default-model").onclick = restoreCourseModel;
$("download-report").onclick = downloadReport;
$("clear-trials").onclick = () => { state.trials = []; renderTrials(); };
$("capture-stroke").onclick = captureStroke;
$("undo-capture").onclick = () => { state.captured.strokes.pop(); renderCaptureCounts(); saveCaptureDraft(); setStatus("Removed the most recently captured virtual stroke. The recovery draft was updated."); };
$("review-capture").onclick = () => { const stroke = state.captured.strokes[Number($("capture-select").value)]; if (!stroke) return; $("actual-label").value = stroke.label; state.currentSource = `capture review · ${stroke.id}`; setPoints(stroke.points); setStatus(`Loaded ${stroke.id} for review. Removing it will update the saved draft immediately.`); };
$("remove-capture").onclick = () => { const index = Number($("capture-select").value); const [removed] = state.captured.strokes.splice(index, 1); if (!removed) return; renderCaptureCounts(); saveCaptureDraft(); setStatus(`Removed ${removed.id}. The remaining ${state.captured.strokes.length} examples are autosaved.`); };
$("clear-capture").onclick = () => { if (!window.confirm("Clear every autosaved capture in this browser? Export first if you may need this work.")) return; state.captured = emptyCapturedDataset(); localStorage.removeItem(CAPTURE_STORAGE_KEY); renderCaptureCounts(); $("capture-storage-status").textContent = "Saved draft cleared. New captures will autosave again."; setStatus("Capture draft cleared. This cannot be undone unless you previously exported the dataset JSON."); };
$("export-dataset").onclick = exportCapturedDataset;
for (const id of ["confidence", "margin"]) $(id).oninput = () => { $(`${id}-value`).textContent = Number($(id).value).toFixed(2); };

$("model-file").onchange = async (event) => { try { const file = event.target.files[0]; if (file) await loadModel(await file.arrayBuffer(), file.name, null); } catch (error) { setStatus(`Model rejected: ${error.message}`, true); } };
$("metrics-file").onchange = async (event) => { try { const file = event.target.files[0]; if (!file) return; const metrics = JSON.parse(await file.text()); if (!Array.isArray(metrics.labels) || metrics.labels.join(",") !== LABELS.join(",") || !metrics.tensorMetadata?.inputQuantization || !metrics.tensorMetadata?.outputQuantization) throw new Error("metrics.json does not contain the course labels and quantization metadata"); if (state.modelHash && metrics.modelSha256 && metrics.modelSha256 !== state.modelHash) throw new Error("metrics.json belongs to a different model SHA-256"); state.metrics = metrics; renderContract(); drawRaster(); setStatus("Metrics loaded. Quantization and operator evidence now follow the learner export."); } catch (error) { setStatus(`Metrics rejected: ${error.message}`, true); } };
$("dataset-file").onchange = async (event) => { try { const file = event.target.files[0]; if (!file) return; const dataset = JSON.parse(await file.text()); const audit = validateDataset(dataset); if (!audit.valid) throw new Error(audit.errors.slice(0, 3).join("; ")); state.dataset = dataset; populateDataset(); if (dataset.boardProfile === "virtual-wand-v1") { state.captured = { ...dataset, exportedAt: null, sessions: structuredClone(dataset.sessions), strokes: structuredClone(dataset.strokes) }; renderCaptureCounts(); saveCaptureDraft("Imported dataset restored as the editable capture draft."); } } catch (error) { setStatus(`Dataset rejected: ${error.message}`, true); } };

configureLabContext(); setPoints([]); renderTrials(); const restoredCaptureMessage = restoreCaptureDraft(); renderCaptureCounts(); $("capture-storage-status").textContent = restoredCaptureMessage; restoreCourseModel();
