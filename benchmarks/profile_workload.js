#!/usr/bin/env node
/**
 * Stage 7 — a fixed workload to profile with V8's sampling profiler:
 *
 *   node --cpu-prof --cpu-prof-dir=benchmarks/results/profiles --cpu-prof-interval 500 benchmarks/profile_workload.js
 *   node benchmarks/analyze_profile.js benchmarks/results/profiles/<file>.cpuprofile
 *
 * Workload = what the app does: `ITER` full distributed generations of "MESH" + 10 tokens through the real PipelineA/PipelineB.
 * The sampling profiler interrupts the program every `interval` microseconds and records the current call stack. Functions that
 * appear in many samples are where the time goes. It is statistical: more samples = more trustworthy percentages.
 */
"use strict";
const { loadModel, PROMPT, GEN_TOKENS, REFERENCE_GENERATION } = require("./lib/model");
const { runDistributed } = require("./lib/runners");
(async () => {
  const model = loadModel();
  const iters = Number(process.argv[2] || 3);
  const rec = await runDistributed(model, { kind: "generate", prompt: PROMPT, newTokens: GEN_TOKENS, expectedTokens: REFERENCE_GENERATION }, { warmup: 1, iterations: iters, maxMessageSize: 262144 });
  if (rec.failureCount) { console.error("profiling workload failed", rec.errors); process.exit(1); }
  console.log(`profiled ${iters} generations (+1 warm-up), failures: 0`);
})();
