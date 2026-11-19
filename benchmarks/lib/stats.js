/**
 * Stage 7 — descriptive statistics for benchmark samples.
 *
 * Why these numbers (see BENCHMARKS.md, "Methodology"):
 *   median  : the typical run. Robust to a single GC pause or OS hiccup, which can inflate the mean a lot.
 *   mean    : kept because totals = mean * n, and because mean >> median is itself a warning sign.
 *   p95     : "tail latency". Linear interpolation between ranks (same as NumPy's default). With n < 20 the p95 is
 *             basically the second-largest sample: treat it as indicative, not precise. The report prints n next to it.
 *   stddev / cv : spread. cv = stddev / mean. A large cv means the measurement itself is noisy and differences
 *             smaller than about 2*cv are not trustworthy.
 *
 * FAILED runs never enter these statistics: a Recorder keeps ok samples and errors in separate lists.
 */
"use strict";

function percentile(sorted, p) {
  if (sorted.length === 0) return NaN;
  if (sorted.length === 1) return sorted[0];
  const rank = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(rank), hi = Math.ceil(rank);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo);
}

function summarize(samples) {
  const xs = Array.from(samples);
  for (const x of xs) if (!Number.isFinite(x)) throw new TypeError("summarize(): non-finite sample " + x);
  const n = xs.length;
  if (n === 0) return { n: 0, mean: NaN, median: NaN, p95: NaN, min: NaN, max: NaN, stddev: NaN, cv: NaN };
  const sorted = xs.slice().sort((a, b) => a - b);
  const mean = xs.reduce((s, x) => s + x, 0) / n;
  const variance = n > 1 ? xs.reduce((s, x) => s + (x - mean) * (x - mean), 0) / (n - 1) : 0;   // sample variance (n-1)
  const stddev = Math.sqrt(variance);
  return { n, mean, median: percentile(sorted, 50), p95: percentile(sorted, 95), min: sorted[0], max: sorted[n - 1], stddev, cv: mean !== 0 ? stddev / mean : 0 };
}

/** Collects named metrics per iteration. Failures are recorded separately and never reach summarize(). */
class Recorder {
  constructor() { this.rows = []; this.errors = []; this.attempts = 0; }
  ok(row) { this.attempts++; this.rows.push(row); }
  fail(err, context = {}) { this.attempts++; this.errors.push({ code: err && err.code ? String(err.code) : "ERROR", message: err && err.message ? String(err.message) : String(err), ...context }); }
  metric(name) { return this.rows.map((r) => r[name]).filter((x) => x !== undefined && x !== null); }
  summary(names) { const out = {}; for (const n of names) out[n] = summarize(this.metric(n)); return out; }
  get failureCount() { return this.errors.length; }
}

/** Absolute and percentage difference of b relative to a (a is the baseline). */
function compare(a, b) {
  return { baseline: a, other: b, absDiff: b - a, pctDiff: a !== 0 ? ((b - a) / a) * 100 : NaN, ratio: a !== 0 ? b / a : NaN };
}

/**
 * Bootstrap confidence interval for (median(b) - median(a)). Resample each sample with replacement `B` times and look at the spread
 * of the difference of medians. If the 95% interval contains 0 we CANNOT claim a and b differ: the gap is within measurement noise.
 * Deterministic (seeded) so a report can be reproduced.
 */
function bootstrapMedianDiffCI(a, b, { B = 2000, seed = 12345, level = 0.95 } = {}) {
  if (a.length === 0 || b.length === 0) return { lo: NaN, hi: NaN, excludesZero: false };
  let s = seed >>> 0;
  const rnd = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
  const med = (xs) => { const t = xs.slice().sort((p, q) => p - q); const m = t.length >> 1; return t.length % 2 ? t[m] : (t[m - 1] + t[m]) / 2; };
  const draw = (xs) => xs.map(() => xs[Math.floor(rnd() * xs.length)]);
  const diffs = [];
  for (let i = 0; i < B; i++) diffs.push(med(draw(b)) - med(draw(a)));
  diffs.sort((p, q) => p - q);
  const lo = percentile(diffs, ((1 - level) / 2) * 100), hi = percentile(diffs, (1 - (1 - level) / 2) * 100);
  return { lo, hi, excludesZero: lo > 0 || hi < 0, level };
}

/** Concatenate several Recorders (used when a benchmark runs in interleaved rounds). */
function mergeRecorders(list) {
  const r = new Recorder();
  for (const x of list) { r.rows.push(...x.rows); r.errors.push(...x.errors); r.attempts += x.attempts; }
  return r;
}

module.exports = { percentile, summarize, Recorder, compare, bootstrapMedianDiffCI, mergeRecorders };
