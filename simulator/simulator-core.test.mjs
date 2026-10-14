import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { classifyScores, normalizeStroke, quantizeRaster, rasterizeStroke, summarizeTrials, validateDataset } from "./simulator-core.js";

test("raster output matches the Python golden fixture", () => {
  const image = rasterizeStroke([{ x: -0.5, y: 0 }, { x: 0, y: 0.5 }, { x: 0.5, y: 0 }]);
  assert.equal(image.length, 32 * 32 * 3);
  assert.equal(createHash("sha256").update(image).digest("hex"), "c022edc7a57c5ae8fcd43f4a7c7eebd0bd72409dfbbb279fdeafdb242252b9ba");
  const quantized = quantizeRaster(image);
  assert.equal(Math.min(...quantized), -128);
  assert.equal(Math.max(...quantized), 127);
});

test("normalization centers a drawn stroke and keeps it bounded", () => {
  const normalized = normalizeStroke([{ x: 4, y: 1 }, { x: 8, y: 9 }, { x: 12, y: 1 }]);
  assert.ok(Math.abs(normalized.reduce((sum, point) => sum + point.x, 0)) < 1e-9);
  assert.ok(Math.abs(normalized.reduce((sum, point) => sum + point.y, 0)) < 1e-9);
  assert.ok(Math.max(...normalized.flatMap((point) => [Math.abs(point.x), Math.abs(point.y)])) <= 0.6);
});

test("confidence and margin gates reject uncertain named gestures", () => {
  const accepted = classifyScores([102, -100, -110, -120]);
  assert.equal(accepted.predictedLabel, "wing");
  assert.equal(accepted.accepted, true);
  const rejected = classifyScores([35, 25, -100, -100]);
  assert.equal(rejected.predictedLabel, "unknown");
  assert.equal(rejected.accepted, false);
});

test("dataset validation and report summary preserve the four-class contract", () => {
  const points = Array.from({ length: 20 }, (_, index) => ({ x: index / 20, y: 0, t: index / 19 }));
  const dataset = { schemaVersion: 1, boardProfile: "virtual-wand-v1", sessions: [{ id: "s1", label: "wing", split: "test" }], strokes: [{ id: "g1", label: "wing", sessionId: "s1", points }] };
  assert.equal(validateDataset(dataset).valid, true);
  const summary = summarizeTrials([{ actualLabel: "wing", predictedLabel: "wing" }, { actualLabel: "unknown", predictedLabel: "ring" }]);
  assert.equal(summary.accuracy, 0.5);
  assert.equal(summary.unknownFalseAcceptance, 1);
});
