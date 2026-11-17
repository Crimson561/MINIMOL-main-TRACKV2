#!/usr/bin/env node
/**
 * Stage 7 — before/after for ONE optimization, measured fairly on a noisy machine.
 *   BEFORE = benchmarks/baseline/inference_engine.baseline.js (the engine at git HEAD, unmodified)
 *   AFTER  = engine/inference_engine.js (current)
 * Both are loaded into the SAME process, with the SAME weights files, and run interleaved (ABBA) so that machine drift,
 * thermal changes and JIT state affect both equally. Local (complete model) and distributed (PipelineA/B over loopback) are both measured.
 *   node benchmarks/optimize_compare.js <label> [rounds]
 */
"use strict";
const fs = require("fs"), path = require("path");
const { loadModel, PROMPT, GEN_TOKENS, REFERENCE_GENERATION, fixedTokens, ROOT } = require("./lib/model");
const { runLocal, runDistributed } = require("./lib/runners");
const { summarize, bootstrapMedianDiffCI, mergeRecorders } = require("./lib/stats");
const { describeEnvironment } = require("./lib/env");

(async () => {
  const label = process.argv[2] || "opt"; const rounds = Number(process.argv[3] || 4);
  const beforeFile = process.argv[4] || "inference_engine.baseline.js";   // which saved engine is the BEFORE
  const only = process.argv[5] ? process.argv[5].split(",") : null;        // optional workload ids, e.g. fwd512,fwd128
  const before = loadModel(require("./baseline/" + beforeFile));
  const after = loadModel();
  const plans = [ { id: "fwd8", T: 8, warm: 3, it: 3 }, { id: "fwd35", T: 35, warm: 3, it: 3 }, { id: "fwd128", T: 128, warm: 2, it: 2 }, { id: "fwd512", T: 512, warm: 1, it: 2 }, { id: "gen", gen: true, warm: 2, it: 2 } ].filter((p) => !only || only.includes(p.id));
  const out = { schema: "minimol-stage7-optimization/1", label, beforeEngine: beforeFile, environment: describeEnvironment(ROOT), evidenceLevel: "node-simulation-loopback", rounds, workloads: [] };
  for (const p of plans) {
    const wl = p.gen ? { kind: "generate", prompt: PROMPT, newTokens: GEN_TOKENS, expectedTokens: REFERENCE_GENERATION }
      : (() => { const tokens = fixedTokens(p.T); return { kind: "forward", tokens, expectedToken: before.full.forward(tokens).nextByte }; })();
    const rec = { localBefore: [], localAfter: [], distBefore: [], distAfter: [] };
    for (let r = 0; r < rounds; r++) {
      const o = { warmup: r === 0 ? p.warm : 0, iterations: p.it };
      const steps = [["localBefore", () => runLocal(before, wl, o)], ["localAfter", () => runLocal(after, wl, o)], ["distBefore", () => runDistributed(before, wl, { ...o, maxMessageSize: 262144 })], ["distAfter", () => runDistributed(after, wl, { ...o, maxMessageSize: 262144 })]];
      if (r % 2) steps.reverse();
      for (const [k, fn] of steps) rec[k].push(await fn());
    }
    const M = Object.fromEntries(Object.entries(rec).map(([k, v]) => [k, mergeRecorders(v)]));
    const row = { id: p.id, failures: Object.fromEntries(Object.entries(M).map(([k, v]) => [k, v.errors.length])) };
    for (const mode of ["local", "dist"]) {
      const b = M[mode + "Before"].metric("totalMs"), a = M[mode + "After"].metric("totalMs");
      const sb = summarize(b), sa = summarize(a), ci = bootstrapMedianDiffCI(b, a);
      row[mode] = { n: [b.length, a.length], beforeMedianMs: sb.median, afterMedianMs: sa.median, beforeMinMs: sb.min, afterMinMs: sa.min, absDiffMs: sa.median - sb.median, pctDiff: ((sa.median - sb.median) / sb.median) * 100, speedup: sb.median / sa.median, ci95: ci, beforeCv: sb.cv, afterCv: sa.cv, raw: { before: b, after: a } };
    }
    if (p.gen) { row.tokensPerSec = { localBefore: summarize(M.localBefore.metric("tokensPerSec")).median, localAfter: summarize(M.localAfter.metric("tokensPerSec")).median, distBefore: summarize(M.distBefore.metric("tokensPerSec")).median, distAfter: summarize(M.distAfter.metric("tokensPerSec")).median }; }
    row.distAfterProtocolMedianMs = summarize(M.distAfter.metric("protocolMs")).median; row.distBeforeProtocolMedianMs = summarize(M.distBefore.metric("protocolMs")).median;
    out.workloads.push(row);
    console.log(`[opt] ${p.id}: local ${row.local.beforeMedianMs.toFixed(1)} -> ${row.local.afterMedianMs.toFixed(1)} ms (${row.local.pctDiff.toFixed(1)}%, x${row.local.speedup.toFixed(2)}, ${row.local.ci95.excludesZero ? "significant" : "NOISE"}) | dist ${row.dist.beforeMedianMs.toFixed(1)} -> ${row.dist.afterMedianMs.toFixed(1)} ms (${row.dist.pctDiff.toFixed(1)}%, x${row.dist.speedup.toFixed(2)}, ${row.dist.ci95.excludesZero ? "significant" : "NOISE"}) | failures ${JSON.stringify(row.failures)}`);
  }
  fs.writeFileSync(path.join(__dirname, "results", `${label}.before_after.json`), JSON.stringify(out, null, 2));
  process.exitCode = out.workloads.some((w) => Object.values(w.failures).some((x) => x)) ? 1 : 0;
})().catch((e) => { console.error(e); process.exit(2); });
