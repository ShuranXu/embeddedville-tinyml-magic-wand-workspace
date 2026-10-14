export const LABELS = ["wing", "ring", "slope", "unknown"];
export const IMAGE_SIZE = 32;
export const SOURCE_PROFILES = ["original-lsm9ds1", "rev2-bmi270", "virtual-wand-v1", "fixture"];

function roundHalfEven(value) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

export function normalizeStroke(points) {
  if (!Array.isArray(points) || points.length < 2) return [];
  const meanX = points.reduce((sum, point) => sum + Number(point.x), 0) / points.length;
  const meanY = points.reduce((sum, point) => sum + Number(point.y), 0) / points.length;
  const centered = points.map((point) => ({ x: Number(point.x) - meanX, y: Number(point.y) - meanY }));
  const extent = Math.max(...centered.flatMap((point) => [Math.abs(point.x), Math.abs(point.y)]));
  if (!Number.isFinite(extent) || extent === 0) return centered.map((_, index) => ({ x: 0, y: 0, t: index / Math.max(1, centered.length - 1) }));
  return centered.map((point, index) => ({
    x: point.x / extent * 0.6,
    y: point.y / extent * 0.6,
    t: index / Math.max(1, centered.length - 1),
  }));
}

export function rasterizeStroke(points) {
  const image = new Uint8Array(IMAGE_SIZE * IMAGE_SIZE * 3);
  const count = points.length;
  for (let index = 0; index < count - 1; index += 1) {
    const start = points[index];
    const end = points[index + 1];
    const x0 = roundHalfEven((start.x / 0.6 + 1) * IMAGE_SIZE / 2);
    const y0 = roundHalfEven((-start.y / 0.6 + 1) * IMAGE_SIZE / 2);
    const x1 = roundHalfEven((end.x / 0.6 + 1) * IMAGE_SIZE / 2);
    const y1 = roundHalfEven((-end.y / 0.6 + 1) * IMAGE_SIZE / 2);
    const time = index / Math.max(1, count);
    const color = time < 0.5
      ? [Math.trunc((1 - 2 * time) * 255), Math.trunc(2 * time * 255), 0]
      : [0, Math.trunc((2 - 2 * time) * 255), Math.trunc((2 * time - 1) * 255)];
    const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
    for (let step = 0; step <= steps; step += 1) {
      const x = roundHalfEven(x0 + (x1 - x0) * step / steps);
      const y = roundHalfEven(y0 + (y1 - y0) * step / steps);
      if (x >= 0 && x < IMAGE_SIZE && y >= 0 && y < IMAGE_SIZE) {
        const offset = (y * IMAGE_SIZE + x) * 3;
        image[offset] = color[0]; image[offset + 1] = color[1]; image[offset + 2] = color[2];
      }
    }
  }
  return image;
}

export function quantizeRaster(raster, quantization = [1, -128]) {
  const [scale, zeroPoint] = quantization;
  if (!(scale > 0)) throw new Error("Input quantization scale must be positive");
  return Int32Array.from(raster, (value) => Math.max(-128, Math.min(127, Math.round(value / scale + zeroPoint))));
}

export function classifyScores(rawScores, labels = LABELS, quantization = [0.00390625, -128], thresholds = { confidence: 0.65, margin: 0.15 }) {
  if (rawScores.length !== labels.length) throw new Error(`Expected ${labels.length} output scores, received ${rawScores.length}`);
  const [scale, zeroPoint] = quantization;
  const scores = Array.from(rawScores, (value) => (Number(value) - zeroPoint) * scale);
  const ranked = scores.map((score, index) => ({ label: labels[index], score })).sort((a, b) => b.score - a.score);
  const confidence = ranked[0].score;
  const margin = confidence - ranked[1].score;
  const accepted = ranked[0].label !== "unknown" && confidence >= thresholds.confidence && margin >= thresholds.margin;
  return { scores, topLabel: ranked[0].label, predictedLabel: accepted ? ranked[0].label : "unknown", confidence, margin, accepted };
}

export function validateDataset(data) {
  const errors = [];
  if (!data || data.schemaVersion !== 1) errors.push("schemaVersion must equal 1");
  if (!SOURCE_PROFILES.includes(data?.boardProfile)) errors.push("boardProfile must identify a supported physical or virtual source");
  if (!Array.isArray(data?.sessions)) errors.push("sessions must be an array");
  if (!Array.isArray(data?.strokes)) errors.push("strokes must be an array");
  const sessions = new Map();
  for (const session of data?.sessions ?? []) {
    if (!session?.id || sessions.has(session.id)) errors.push(`session ${session?.id ?? "?"} has a missing or duplicate id`);
    if (!LABELS.includes(session?.label) || !["train", "validation", "test"].includes(session?.split)) errors.push(`session ${session?.id ?? "?"} has invalid label or split metadata`);
    sessions.set(session.id, session);
  }
  for (const stroke of data?.strokes ?? []) {
    if (!LABELS.includes(stroke.label)) errors.push(`stroke ${stroke.id ?? "?"} has an unknown label`);
    if (!sessions.has(stroke.sessionId)) errors.push(`stroke ${stroke.id ?? "?"} references a missing session`);
    else if (sessions.get(stroke.sessionId).label !== stroke.label) errors.push(`stroke ${stroke.id ?? "?"} does not match its session label`);
    if (!Array.isArray(stroke.points) || stroke.points.length < 20 || stroke.points.length > 160) errors.push(`stroke ${stroke.id ?? "?"} must contain 20–160 points`);
  }
  return { valid: errors.length === 0, errors, sessionCount: sessions.size, strokeCount: data?.strokes?.length ?? 0 };
}

export function confusionMatrix(trials, labels = LABELS) {
  const matrix = labels.map(() => labels.map(() => 0));
  for (const trial of trials) {
    const actual = labels.indexOf(trial.actualLabel);
    const predicted = labels.indexOf(trial.predictedLabel);
    if (actual >= 0 && predicted >= 0) matrix[actual][predicted] += 1;
  }
  return matrix;
}

export function summarizeTrials(trials, labels = LABELS) {
  const matrix = confusionMatrix(trials, labels);
  const correct = trials.filter((trial) => trial.actualLabel === trial.predictedLabel).length;
  const unknownTrials = trials.filter((trial) => trial.actualLabel === "unknown");
  const falseAccepts = unknownTrials.filter((trial) => trial.predictedLabel !== "unknown").length;
  return {
    count: trials.length,
    accuracy: trials.length ? correct / trials.length : 0,
    unknownFalseAcceptance: unknownTrials.length ? falseAccepts / unknownTrials.length : 0,
    confusionMatrix: matrix,
  };
}
