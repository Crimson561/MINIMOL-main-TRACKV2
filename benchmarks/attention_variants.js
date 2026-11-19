#!/usr/bin/env node
/**
 * Stage 7 — Optimization 2 candidate: loop interchange in the attention weighted-sum (softmax(scores) @ V).
 * Original: for d (64) { for j (0..i) { acc += scores[j] * V[j][d] } }   -> inner loop strides 1152 floats between reads
 * Candidate: for j (0..i) { for d (64) { acc[d] += scores[j] * V[j][d] } } -> inner loop reads one V row contiguously
 * Each acc[d] receives the same terms in the same order (j ascending) in double precision, so results must be bit-identical.
 */
"use strict";
const { summarize, bootstrapMedianDiffCI } = require("./lib/stats");
const D_MODEL = 384, D_HEAD = 64, H = 6;
function weighted_orig(qkv, scores, i, headOffset, attnOut) {
  const outOffset = i * D_MODEL + headOffset;
  for (let d = 0; d < D_HEAD; d++) { let valSum = 0.0; for (let j = 0; j <= i; j++) { const vOffset = j * (3 * D_MODEL) + 2 * D_MODEL + headOffset; valSum += scores[j] * qkv[vOffset + d]; } attnOut[outOffset + d] = valSum; }
}
const acc = new Float64Array(D_HEAD);
function weighted_new(qkv, scores, i, headOffset, attnOut) {
  const outOffset = i * D_MODEL + headOffset; acc.fill(0);
  for (let j = 0; j <= i; j++) { const s = scores[j], v = j * (3 * D_MODEL) + 2 * D_MODEL + headOffset; for (let d = 0; d < D_HEAD; d++) acc[d] += s * qkv[v + d]; }
  for (let d = 0; d < D_HEAD; d++) attnOut[outOffset + d] = acc[d];
}
function rnd(len, seed) { const a = new Float32Array(len); let s = seed; for (let i = 0; i < len; i++) { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; a[i] = (s / 4294967296 - 0.5); } return a; }
function runAll(fn, qkv, scores, T, out) { for (let h = 0; h < H; h++) for (let i = 0; i < T; i++) fn(qkv, scores, i, h * D_HEAD, out); }
if (require.main === module) {
  for (const T of [35, 128, 512]) {
    const qkv = rnd(T * 3 * D_MODEL, 1), scores = rnd(T, 2).map((x) => Math.abs(x)); const o1 = new Float32Array(T * D_MODEL), o2 = new Float32Array(T * D_MODEL);
    runAll(weighted_orig, qkv, scores, T, o1); runAll(weighted_new, qkv, scores, T, o2);
    const same = Buffer.compare(Buffer.from(o1.buffer), Buffer.from(o2.buffer)) === 0;
    const t = { orig: [], new: [] }; const reps = T >= 512 ? 6 : 14;
    for (let r = 0; r < 3; r++) { runAll(weighted_orig, qkv, scores, T, o1); runAll(weighted_new, qkv, scores, T, o2); }
    for (let r = 0; r < reps; r++) for (const k of (r % 2 ? ["new", "orig"] : ["orig", "new"])) { const s = performance.now(); runAll(k === "orig" ? weighted_orig : weighted_new, qkv, scores, T, k === "orig" ? o1 : o2); t[k].push(performance.now() - s); }
    const a = summarize(t.orig), b = summarize(t.new), ci = bootstrapMedianDiffCI(t.orig, t.new);
    console.log(`T=${T}: bit-identical=${same}  orig ${a.median.toFixed(2)} ms -> new ${b.median.toFixed(2)} ms (${(((b.median - a.median) / a.median) * 100).toFixed(1)}%, x${(a.median / b.median).toFixed(2)}, ${ci.excludesZero ? "significant" : "within noise"})`);
  }
}
module.exports = { weighted_orig, weighted_new };
