#!/usr/bin/env node
/**
 * Stage 7 — Optimization 1 candidate kernels for matmulTransposeB (C[M,N] = A[M,K] @ B[N,K]^T + bias).
 * Each candidate must (1) be BIT-IDENTICAL to the original and (2) be faster when timed interleaved (ABBA) against it.
 * "Bit-identical" holds when every output element is still summed as bias, then k = 0..K-1 in order, in double precision.
 */
"use strict";
const { summarize, bootstrapMedianDiffCI } = require("./lib/stats");
const base = require("./baseline/inference_engine.baseline.js");
const original = base.matmulTransposeB || null;

function orig(A, B, bias, M, K, N, out) {
  const result = out || new Float32Array(M * N);
  for (let m = 0; m < M; m++) { const aO = m * K, cO = m * N;
    for (let n = 0; n < N; n++) { const bO = n * K; let sum = bias ? bias[n] : 0.0; for (let k = 0; k < K; k++) sum += A[aO + k] * B[bO + k]; result[cO + n] = sum; } }
  return result;
}
// V1: 4 output columns at once (4 independent accumulators share each A[k] load)
function v1(A, B, bias, M, K, N, out) {
  const result = out || new Float32Array(M * N);
  const N4 = N - (N % 4);
  for (let m = 0; m < M; m++) { const aO = m * K, cO = m * N;
    let n = 0;
    for (; n < N4; n += 4) {
      const b0 = n * K, b1 = b0 + K, b2 = b1 + K, b3 = b2 + K;
      let s0 = bias ? bias[n] : 0.0, s1 = bias ? bias[n + 1] : 0.0, s2 = bias ? bias[n + 2] : 0.0, s3 = bias ? bias[n + 3] : 0.0;
      for (let k = 0; k < K; k++) { const a = A[aO + k]; s0 += a * B[b0 + k]; s1 += a * B[b1 + k]; s2 += a * B[b2 + k]; s3 += a * B[b3 + k]; }
      result[cO + n] = s0; result[cO + n + 1] = s1; result[cO + n + 2] = s2; result[cO + n + 3] = s3;
    }
    for (; n < N; n++) { const bO = n * K; let s = bias ? bias[n] : 0.0; for (let k = 0; k < K; k++) s += A[aO + k] * B[bO + k]; result[cO + n] = s; }
  }
  return result;
}
// V2: 2 rows of A x 4 columns of B (8 accumulators: each B[k] load is reused across 2 rows too)
function v2(A, B, bias, M, K, N, out) {
  const result = out || new Float32Array(M * N);
  const N4 = N - (N % 4); const M2 = M - (M % 2);
  let m = 0;
  for (; m < M2; m += 2) { const a0o = m * K, a1o = a0o + K, c0 = m * N, c1 = c0 + N;
    for (let n = 0; n < N4; n += 4) {
      const b0 = n * K, b1 = b0 + K, b2 = b1 + K, b3 = b2 + K;
      const i0 = bias ? bias[n] : 0.0, i1 = bias ? bias[n + 1] : 0.0, i2 = bias ? bias[n + 2] : 0.0, i3 = bias ? bias[n + 3] : 0.0;
      let p0 = i0, p1 = i1, p2 = i2, p3 = i3, q0 = i0, q1 = i1, q2 = i2, q3 = i3;
      for (let k = 0; k < K; k++) { const x = A[a0o + k], y = A[a1o + k], w0 = B[b0 + k], w1 = B[b1 + k], w2 = B[b2 + k], w3 = B[b3 + k];
        p0 += x * w0; p1 += x * w1; p2 += x * w2; p3 += x * w3; q0 += y * w0; q1 += y * w1; q2 += y * w2; q3 += y * w3; }
      result[c0 + n] = p0; result[c0 + n + 1] = p1; result[c0 + n + 2] = p2; result[c0 + n + 3] = p3;
      result[c1 + n] = q0; result[c1 + n + 1] = q1; result[c1 + n + 2] = q2; result[c1 + n + 3] = q3;
    }
    for (let n = N4; n < N; n++) { const bO = n * K; for (const [ao, co] of [[a0o, c0], [a1o, c1]]) { let s = bias ? bias[n] : 0.0; for (let k = 0; k < K; k++) s += A[ao + k] * B[bO + k]; result[co + n] = s; } }
  }
  for (; m < M; m++) { const aO = m * K, cO = m * N; for (let n = 0; n < N; n++) { const bO = n * K; let s = bias ? bias[n] : 0.0; for (let k = 0; k < K; k++) s += A[aO + k] * B[bO + k]; result[cO + n] = s; } }
  return result;
}
// V3 (Stage 7 optimization 2 candidate): 4 rows x 4 columns per tile = 16 accumulators, A and B loads each reused 4 times
function v3(A, B, bias, M, K, N, out) {
  const result = out || new Float32Array(M * N);
  const N4 = N - (N % 4), M4 = M - (M % 4);
  let m = 0;
  for (; m < M4; m += 4) { const a0 = m * K, a1 = a0 + K, a2 = a1 + K, a3 = a2 + K, c0 = m * N, c1 = c0 + N, c2 = c1 + N, c3 = c2 + N;
    for (let n = 0; n < N4; n += 4) {
      const b0 = n * K, b1 = b0 + K, b2 = b1 + K, b3 = b2 + K;
      const i0 = bias ? bias[n] : 0.0, i1 = bias ? bias[n + 1] : 0.0, i2 = bias ? bias[n + 2] : 0.0, i3 = bias ? bias[n + 3] : 0.0;
      let p0 = i0, p1 = i1, p2 = i2, p3 = i3, q0 = i0, q1 = i1, q2 = i2, q3 = i3, r0 = i0, r1 = i1, r2 = i2, r3 = i3, s0 = i0, s1 = i1, s2 = i2, s3 = i3;
      for (let k = 0; k < K; k++) { const x = A[a0 + k], y = A[a1 + k], z = A[a2 + k], u = A[a3 + k], w0 = B[b0 + k], w1 = B[b1 + k], w2 = B[b2 + k], w3 = B[b3 + k];
        p0 += x * w0; p1 += x * w1; p2 += x * w2; p3 += x * w3; q0 += y * w0; q1 += y * w1; q2 += y * w2; q3 += y * w3;
        r0 += z * w0; r1 += z * w1; r2 += z * w2; r3 += z * w3; s0 += u * w0; s1 += u * w1; s2 += u * w2; s3 += u * w3; }
      result[c0 + n] = p0; result[c0 + n + 1] = p1; result[c0 + n + 2] = p2; result[c0 + n + 3] = p3;
      result[c1 + n] = q0; result[c1 + n + 1] = q1; result[c1 + n + 2] = q2; result[c1 + n + 3] = q3;
      result[c2 + n] = r0; result[c2 + n + 1] = r1; result[c2 + n + 2] = r2; result[c2 + n + 3] = r3;
      result[c3 + n] = s0; result[c3 + n + 1] = s1; result[c3 + n + 2] = s2; result[c3 + n + 3] = s3;
    }
  }
  for (let r = 0; r < M; r++) { const aO = r * K, cO = r * N; for (let n = (r < M4 ? N4 : 0); n < N; n++) { const bO = n * K; let s = bias ? bias[n] : 0.0; for (let k = 0; k < K; k++) s += A[aO + k] * B[bO + k]; result[cO + n] = s; } }
  return result;
}
const VARIANTS = { v2, v3 };

function rnd(len, seed) { const a = new Float32Array(len); let s = seed; for (let i = 0; i < len; i++) { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; a[i] = (s / 4294967296 - 0.5) * 0.2; } return a; }
const same = (x, y) => { const a = new Uint32Array(x.buffer), b = new Uint32Array(y.buffer); for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; };

if (require.main === module) {
  const T = Number(process.argv[2] || 35);
  const shapes = [["qkv", T, 384, 1152], ["attn_proj", T, 384, 384], ["mlp_fc", T, 384, 1536], ["mlp_proj", T, 1536, 384], ["odd shape", 7, 50, 13]];
  console.log("bit-identical check vs original:");
  for (const [name, M, K, N] of shapes) { const A = rnd(M * K, 1), B = rnd(N * K, 2), bias = rnd(N, 3); const ref = orig(A, B, bias, M, K, N);
    console.log("  " + name.padEnd(10), Object.entries(VARIANTS).map(([v, f]) => `${v}:${same(ref, f(A, B, bias, M, K, N))}`).join(" "), "no-bias:", Object.entries(VARIANTS).map(([v, f]) => same(orig(A, B, null, M, K, N), f(A, B, null, M, K, N))).join(" ")); }
  const [name, M, K, N] = shapes[0]; const A = rnd(M * K, 1), B = rnd(N * K, 2), bias = rnd(N, 3), out = new Float32Array(M * N);
  const all = { orig, ...VARIANTS }; const t = Object.fromEntries(Object.keys(all).map((k) => [k, []]));
  for (const f of Object.values(all)) for (let i = 0; i < 20; i++) f(A, B, bias, M, K, N, out);
  for (let r = 0; r < 30; r++) { const order = Object.keys(all); if (r % 2) order.reverse(); for (const k of order) { const s = performance.now(); for (let i = 0; i < 5; i++) all[k](A, B, bias, M, K, N, out); t[k].push((performance.now() - s) / 5); } }
  console.log(`\ninterleaved timing, ${name} T=${T} (ms per call, 30 rounds):`);
  for (const k of Object.keys(all)) { const s = summarize(t[k]); const ci = k === "orig" ? null : bootstrapMedianDiffCI(t.orig, t[k]); console.log(`  ${k.padEnd(5)} median ${s.median.toFixed(3)}  min ${s.min.toFixed(3)}  cv ${s.cv.toFixed(2)}${ci ? `  vs orig ${(((s.median - summarize(t.orig).median) / summarize(t.orig).median) * 100).toFixed(1)}%  CI[${ci.lo.toFixed(3)},${ci.hi.toFixed(3)}] ${ci.excludesZero ? "significant" : "within noise"}` : ""}`); }
}
module.exports = { orig, v1, v2, v3, VARIANTS };
