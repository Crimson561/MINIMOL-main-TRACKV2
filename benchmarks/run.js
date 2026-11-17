#!/usr/bin/env node
/**
 * MINIMOL Stage 7 benchmark runner.
 *
 *   node benchmarks/run.js --profile standard --label baseline
 *   node benchmarks/run.js --profile quick --label smoke
 *
 * Evidence level of everything this script produces: "Node simulation, single machine, loopback link, NO network".
 * It is not a browser run, not a two-tab run and not a physical-device run (see benchmarks/browser/ and BENCHMARKS.md for those).
 */
"use strict";
const fs = require("fs");
const path = require("path");
const { loadModel, PROMPT, GEN_TOKENS, REFERENCE_GENERATION, fixedTokens, ROOT } = require("./lib/model");
const { runLocal, runDistributed } = require("./lib/runners");
const { summarize, compare, bootstrapMedianDiffCI, mergeRecorders } = require("./lib/stats");
const { describeEnvironment } = require("./lib/env");
const { runMicro } = require("./lib/micro");
const { numericalCheck } = require("./lib/numerics");
const { renderMarkdown } = require("./report");

const PROFILES = {
  // rounds: how many interleaved (local,distributed) blocks; iterations: measured runs PER ROUND; warmup: discarded runs, first round only
  quick:    { rounds: 1, workloads: [ { id: "fwd8", T: 8, warmup: 1, iterations: 3 }, { id: "fwd35", T: 35, warmup: 1, iterations: 3 }, { id: "gen", gen: true, warmup: 1, iterations: 2 } ], micro: { sizes: [8, 35], iterations: 15, warmup: 5 }, numericsMaxT: 31 },
  standard: { rounds: 3, workloads: [ { id: "fwd8", T: 8, warmup: 3, iterations: 5 }, { id: "fwd35", T: 35, warmup: 3, iterations: 5 }, { id: "fwd128", T: 128, warmup: 2, iterations: 3 }, { id: "fwd512", T: 512, warmup: 1, iterations: 1 }, { id: "gen", gen: true, warmup: 2, iterations: 4 } ], micro: {}, numericsMaxT: Infinity },
};

function arg(name, def) { const i = process.argv.indexOf("--" + name); return i > 0 ? process.argv[i + 1] : def; }

function cpuSnap() { return { cpu: process.cpuUsage(), t: process.hrtime.bigint() }; }
function cpuUtil(a, b) { const cpuMs = ((b.cpu.user - a.cpu.user) + (b.cpu.system - a.cpu.system)) / 1000; const wallMs = Number(b.t - a.t) / 1e6; return { cpuMs, wallMs }; }

async function main() {
  const profileName = arg("profile", "standard"), label = arg("label", "run");
  const profile = PROFILES[profileName]; if (!profile) throw new Error("unknown profile " + profileName);
  const t0 = Date.now();
  const mem0 = process.memoryUsage();
  const model = loadModel();
  const mem1 = process.memoryUsage();
  const out = {
    schema: "minimol-stage7-benchmark/1", label, profile: profileName,
    evidenceLevel: "node-simulation-loopback (single machine, no network, no browser, no second physical device)",
    environment: describeEnvironment(ROOT), model: model.info,
    memory: { afterLoadWeights: { rssMB: +(mem1.rss / 1048576).toFixed(1), heapUsedMB: +(mem1.heapUsed / 1048576).toFixed(1), arrayBuffersMB: +(mem1.arrayBuffers / 1048576).toFixed(1) }, beforeLoad: { rssMB: +(mem0.rss / 1048576).toFixed(1) },
      note: "weights are mapped as Float32Array views over the loaded ArrayBuffers; per-device weight memory = partition file size" },
    config: { rounds: profile.rounds, workloads: profile.workloads, prompt: PROMPT, generatedTokens: GEN_TOKENS, channelMaxMessageSize: 262144, order: "ABBA: local first in even rounds, distributed first in odd rounds" },
    numerics: null, micro: null, comparisons: [],
  };

  console.log(`[bench] numerical gate...`);
  out.numerics = await numericalCheck(model, { maxT: profile.numericsMaxT });
  console.log(`[bench]   passed=${out.numerics.passed} maxAbs(PyTorch) act=${out.numerics.maxAbsPyTorchActivation.toExponential(2)} logits=${out.numerics.maxAbsPyTorchLogits.toExponential(2)} digest=${out.numerics.logitsDigestAdler32}`);

  console.log(`[bench] communication micro-benchmarks...`);
  out.micro = runMicro(profile.micro);

  for (const w of profile.workloads) {
    const workload = w.gen ? { kind: "generate", prompt: PROMPT, newTokens: GEN_TOKENS, expectedTokens: REFERENCE_GENERATION }
      : (() => { const tokens = fixedTokens(w.T); return { kind: "forward", tokens, expectedToken: model.full.forward(tokens).nextByte }; })();
    const locals = [], dists = []; const cpu = { local: { cpuMs: 0, wallMs: 0 }, distributed: { cpuMs: 0, wallMs: 0 } };
    for (let r = 0; r < profile.rounds; r++) {
      const warm = r === 0 ? w.warmup : 0, opts = { warmup: warm, iterations: w.iterations };
      const doLocal = () => { const a = cpuSnap(); locals.push(runLocal(model, workload, opts)); const u = cpuUtil(a, cpuSnap()); cpu.local.cpuMs += u.cpuMs; cpu.local.wallMs += u.wallMs; };
      const doDist = async () => { const a = cpuSnap(); dists.push(await runDistributed(model, workload, { ...opts, maxMessageSize: 262144 })); const u = cpuUtil(a, cpuSnap()); cpu.distributed.cpuMs += u.cpuMs; cpu.distributed.wallMs += u.wallMs; };
      if (r % 2 === 0) { doLocal(); await doDist(); } else { await doDist(); doLocal(); }
    }
    const L = mergeRecorders(locals), D = mergeRecorders(dists);
    const lt = L.metric("totalMs"), dt = D.metric("totalMs");
    const cmp = compare(summarize(lt).median, summarize(dt).median);
    const ci = bootstrapMedianDiffCI(lt, dt);
    const protoMed = summarize(D.metric("protocolMs")).median;
    const entry = {
      id: w.id, kind: workload.kind, T: workload.kind === "forward" ? w.T : null, generation: workload.kind === "generate" ? { promptTokens: PROMPT.length, newTokens: GEN_TOKENS } : null,
      attempts: { local: L.attempts, distributed: D.attempts }, failures: { local: L.errors, distributed: D.errors },
      local: L.summary(L.rows[0] ? Object.keys(L.rows[0]) : []), distributed: D.summary(D.rows[0] ? Object.keys(D.rows[0]) : []),
      rawTotalMs: { local: L.metric("totalMs"), distributed: D.metric("totalMs"), note: "samples in execution order across the interleaved rounds" },
      cpuUtilization: { local: cpu.local.cpuMs / cpu.local.wallMs, distributed: cpu.distributed.cpuMs / cpu.distributed.wallMs },
      comparison: { medianLocalMs: cmp.baseline, medianDistributedMs: cmp.other, absDiffMs: cmp.absDiff, pctDiff: cmp.pctDiff, ratio: cmp.ratio, bootstrapCI95: ci,
        medianProtocolOverheadMs: protoMed, protocolOverheadPctOfDistributed: (protoMed / cmp.other) * 100 },
    };
    if (workload.kind === "generate") {
      const lt2 = summarize(L.metric("tokensPerSec")).median, dt2 = summarize(D.metric("tokensPerSec")).median;
      entry.comparison.tokensPerSecLocal = lt2; entry.comparison.tokensPerSecDistributed = dt2; entry.comparison.tokensPerSecDiffPct = ((dt2 - lt2) / lt2) * 100;
    }
    const MIN_N = 8;   // below this the bootstrap interval is meaningless; refuse to issue a verdict
    const verdict = Math.min(lt.length, dt.length) < MIN_N ? `no verdict: only ${Math.min(lt.length, dt.length)} samples per side (need >= ${MIN_N})` : !ci.excludesZero ? "no measurable difference (95% CI of the median difference contains 0)" : cmp.absDiff > 0 ? "distributed SLOWER (difference is outside measurement noise)" : "distributed FASTER (difference is outside measurement noise)";
    entry.comparison.verdict = verdict;
    out.comparisons.push(entry);
    console.log(`[bench] ${w.id}: local ${cmp.baseline.toFixed(1)} ms | distributed ${cmp.other.toFixed(1)} ms | diff ${cmp.absDiff.toFixed(1)} ms (${cmp.pctDiff.toFixed(1)}%) | protocol ${protoMed.toFixed(2)} ms | failures L=${L.errors.length} D=${D.errors.length} | ${verdict}`);
  }
  out.peakRssMB = +(process.resourceUsage().maxRSS / 1024).toFixed(1);
  out.wallClockSeconds = +((Date.now() - t0) / 1000).toFixed(1);

  const dir = path.join(__dirname, "results"); fs.mkdirSync(dir, { recursive: true });
  const base = path.join(dir, `${label}.${profileName}`);
  fs.writeFileSync(base + ".json", JSON.stringify(out, null, 2));
  fs.writeFileSync(base + ".md", renderMarkdown(out));
  // flat CSV for spreadsheets: one row per workload per mode, medians only
  const rows = ["workload,mode,n_ok,n_failed,median_ms,mean_ms,p95_ms,min_ms,max_ms,stddev_ms,cv"];
  for (const c of out.comparisons) for (const m of ["local", "distributed"]) { const s = c[m].totalMs; rows.push([c.id, m, s.n, c.failures[m].length, s.median, s.mean, s.p95, s.min, s.max, s.stddev, s.cv].map((x) => typeof x === "number" ? +x.toFixed(4) : x).join(",")); }
  fs.writeFileSync(base + ".csv", rows.join("\n") + "\n");
  console.log(`[bench] wrote ${path.relative(ROOT, base)}.{json,md,csv} in ${out.wallClockSeconds}s`);
  const anyFail = out.comparisons.some((c) => c.failures.local.length || c.failures.distributed.length) || !out.numerics.passed;
  process.exitCode = anyFail ? 1 : 0;
}
main().catch((e) => { console.error(e); process.exit(2); });
