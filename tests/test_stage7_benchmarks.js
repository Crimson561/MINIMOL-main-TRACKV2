/**
 * Stage 7 tests: the benchmark tooling is correct, and the matmul optimization preserved numerics.
 * Level of evidence: unit tests and Node simulations (loopback, no network). These verify the MEASURING INSTRUMENTS and the optimization's
 * correctness. They do not assert any speed number (speed on a shared machine is noisy), and they are not browser or device benchmarks.
 *   node tests/test_stage7_benchmarks.js
 */
"use strict";
const path = require("path");
const fs = require("fs");
const { execFileSync } = require("child_process");
const B = path.join(__dirname, "..", "benchmarks");
const E = require("../engine/inference_engine");
const P = require("../protocol/mesh_protocol");
const { summarize, percentile, Recorder, bootstrapMedianDiffCI } = require(B + "/lib/stats");
const { loadModel, fixedTokens, PROMPT, REFERENCE_GENERATION } = require(B + "/lib/model");
const { runLocal, runDistributed, makeRig, distributedRequest } = require(B + "/lib/runners");
const { LoopbackLink } = require(B + "/lib/loopback");
const { runMicro } = require(B + "/lib/micro");
const { numericalCheck } = require(B + "/lib/numerics");
const { orig: naiveMatmul } = require(B + "/matmul_variants");

let passed = 0, failed = 0;
function ok(cond, name) { if (cond) { passed++; console.log("  ✓ " + name); } else { failed++; console.log("  ✗ FAIL: " + name); } }
const close = (a, b, eps) => Math.abs(a - b) <= eps;
const bits = (x, y) => Buffer.compare(Buffer.from(x.buffer, x.byteOffset, x.byteLength), Buffer.from(y.buffer, y.byteOffset, y.byteLength)) === 0;
function rnd(len, seed) { const a = new Float32Array(len); let s = seed; for (let i = 0; i < len; i++) { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; a[i] = (s / 4294967296 - 0.5) * 0.4; } return a; }

(async () => {
  console.log("\n== 1. statistics ==");
  const s = summarize([1, 2, 3, 4, 100]);
  ok(s.n === 5 && s.median === 3 && s.min === 1 && s.max === 100 && close(s.mean, 22, 1e-12), "n, median, min, max, mean on a known sample (median ignores the outlier, mean does not)");
  ok(close(percentile([10, 20, 30, 40], 50), 25, 1e-12) && close(percentile([0, 10], 95), 9.5, 1e-12), "percentile uses linear interpolation between ranks");
  ok(close(summarize([2, 4, 4, 4, 5, 5, 7, 9]).stddev, Math.sqrt(32 / 7), 1e-12), "stddev is the sample (n-1) standard deviation");
  let threw = false; try { summarize([1, NaN]); } catch (e) { threw = true; } ok(threw, "summarize refuses NaN/Infinity instead of silently producing garbage");
  ok(summarize([]).n === 0 && Number.isNaN(summarize([]).median), "empty sample gives n=0 and NaN, not 0");
  const rec = new Recorder(); rec.ok({ totalMs: 10 }); rec.ok({ totalMs: 12 }); rec.fail(Object.assign(new Error("boom"), { code: "X" }), { iteration: 2 });
  ok(rec.metric("totalMs").length === 2 && rec.errors.length === 1 && rec.attempts === 3 && rec.errors[0].code === "X", "a failed attempt is counted and stored in the error list but NEVER enters latency statistics");
  const a = [10, 11, 10.5, 10.2, 10.8, 10.4, 10.9, 10.1], same = [10.1, 10.9, 10.4, 10.6, 10.3, 10.7, 10.2, 10.5], far = a.map((x) => x * 2);
  ok(!bootstrapMedianDiffCI(a, same).excludesZero, "bootstrap CI: two samples from the same distribution are NOT declared different");
  ok(bootstrapMedianDiffCI(a, far).excludesZero, "bootstrap CI: a 2x shift IS declared different");
  ok(JSON.stringify(bootstrapMedianDiffCI(a, far)) === JSON.stringify(bootstrapMedianDiffCI(a, far)), "bootstrap is seeded, so a report can be reproduced");

  console.log("\n== 2. matmul optimization is bit-identical to the naive loop ==");
  for (const [name, M, K, N] of [["qkv T=35", 35, 384, 1152], ["attn proj", 9, 384, 384], ["mlp fc", 4, 384, 1536], ["mlp proj", 3, 1536, 384], ["lm_head M=1", 1, 384, 256], ["odd 7x50x13", 7, 50, 13], ["tiny 1x1x1", 1, 1, 1], ["odd M=5,N=6", 5, 17, 6]]) {
    const A = rnd(M * K, 1), Bm = rnd(N * K, 2), bias = rnd(N, 3);
    ok(bits(naiveMatmul(A, Bm, bias, M, K, N), E.matmulTransposeB(A, Bm, bias, M, K, N)) && bits(naiveMatmul(A, Bm, null, M, K, N), E.matmulTransposeB(A, Bm, null, M, K, N)), `${name}: bit-identical with and without bias`);
  }
  const out = new Float32Array(35 * 384); const r = E.matmulTransposeB(rnd(35 * 384, 4), rnd(384 * 384, 5), null, 35, 384, 384, out); ok(r === out, "the optional `out` buffer is still used and returned");

  console.log("\n== 3. instrumentation (Node loopback rig) ==");
  const model = loadModel();
  const encBefore = P.encodeActivation, splitBefore = P.splitIntoChunks, decBefore = P.decodePacket;
  const rig = await makeRig(model);
  ok(P.encodeActivation !== encBefore, "install() really wraps the protocol functions");
  const tokens = fixedTokens(8), expected = model.full.forward(tokens).nextByte;
  const { row, token } = await distributedRequest(rig, tokens);
  ok(token === expected, "distributed next token equals local next token");
  const spanKeys = ["aCompute", "aSerialize", "aChunk", "aSend", "bReassemble", "bDecode", "bValidate", "bCompute", "bSerializeReply", "bChunkReply", "bSend", "aReassembleReply", "aDecodeReply"];
  ok(spanKeys.every((k) => row[k] >= 0), "every phase span is non-negative");
  ok(row.attributedMs <= row.totalMs + 0.05, "phase spans are disjoint: their sum never exceeds the wall-clock total (no double counting)");
  ok(row.unattributedMs >= -0.05, "unattributed time is reported as total - sum(spans) and is not negative");
  ok(row.aSerialize > 0 && row.bDecode > 0 && row.bValidate >= 0 && row.aCompute > 0 && row.bCompute > 0, "encode, decode, validate and both computes were actually observed");
  ok(Math.abs(row.aCompute - row.pipeComputeMsA) < Math.max(2, 0.05 * row.pipeComputeMsA) && Math.abs(row.bCompute - row.pipeComputeMsB) < Math.max(2, 0.05 * row.pipeComputeMsB), "wrapper compute times agree with the pipeline's own computeMsA/computeMsB");
  ok(row.frames === 1 && row.activationPacketBytes === 48 + 8 * 384 * 4, "packet size = 48-byte header + T*384*4 bytes; one frame at T=8");
  ok(row.protocolMs < row.totalMs * 0.2, "protocol (non-compute) time is a small part of one request at T=8");
  rig.dispose();
  ok(P.encodeActivation === encBefore && P.splitIntoChunks === splitBefore && P.decodePacket === decBefore, "dispose() restores the original protocol functions");

  console.log("\n== 4. multi-frame request is chunked and counted ==");
  const small = await makeRig(model, { maxMessageSize: 16384 });
  const big = await distributedRequest(small, fixedTokens(35));   // 53,808 bytes > 16,384
  ok(big.row.frames === Math.ceil(53808 / (16384 - P.CHUNK_HEADER_SIZE)) && big.row.aChunk > 0 && big.row.bReassemble > 0, `a ${big.row.activationPacketBytes}-byte packet over a 16 KiB limit is split into ${big.row.frames} frames and reassembled`);
  small.dispose();

  console.log("\n== 5. failures are kept out of the statistics ==");
  const bad = await runDistributed(model, { kind: "forward", tokens: fixedTokens(8), expectedToken: (expected + 1) % 256 }, { warmup: 0, iterations: 3, maxMessageSize: 262144 });
  ok(bad.failureCount === 3 && bad.rows.length === 0 && bad.errors[0].code === "OUTPUT_MISMATCH" && summarize(bad.metric("totalMs")).n === 0, "wrong expected output: 3 failures recorded, 0 successful samples, nothing leaks into the stats");
  const badLocal = runLocal(model, { kind: "forward", tokens: fixedTokens(8), expectedToken: (expected + 1) % 256 }, { warmup: 0, iterations: 2 });
  ok(badLocal.failureCount === 2 && badLocal.rows.length === 0, "same for the local path");
  let warmThrew = false; try { await runDistributed(model, { kind: "forward", tokens: fixedTokens(8), expectedToken: (expected + 1) % 256 }, { warmup: 1, iterations: 1, maxMessageSize: 262144 }); } catch (e) { warmThrew = true; }
  ok(warmThrew, "a failing WARM-UP aborts loudly (a broken rig must not produce a report)");
  const good = await runDistributed(model, { kind: "generate", prompt: PROMPT, newTokens: 10, expectedTokens: REFERENCE_GENERATION }, { warmup: 0, iterations: 1, maxMessageSize: 262144 });
  ok(good.failureCount === 0 && good.rows.length === 1, "distributed generation reproduces the PyTorch reference sequence (checked inside the runner)");
  const goodLocal = runLocal(model, { kind: "generate", prompt: PROMPT, newTokens: 10, expectedTokens: REFERENCE_GENERATION }, { warmup: 0, iterations: 1 });
  ok(goodLocal.failureCount === 0, "local generation reproduces the same reference sequence");

  console.log("\n== 6. loopback link has no artificial latency ==");
  const link = new LoopbackLink(1000); let got = null; link.toB = (m) => { got = m; };
  const tr = link.transport("AtoB"); const t0 = performance.now(); const buf = new Uint8Array([1, 2, 3]).buffer; tr.send(buf);
  await new Promise((r) => setImmediate(() => setImmediate(r))); const dt = performance.now() - t0;
  ok(got && got.byteLength === 3 && got !== buf && dt < 5, `a message is delivered as a copy within one event-loop turn (${dt.toFixed(2)} ms; the Stage 5 test Link uses a fixed 1 ms setTimeout)`);
  let over = false; try { tr.send(new ArrayBuffer(2000)); } catch (e) { over = e instanceof RangeError; } ok(over, "a frame larger than the channel limit is rejected, like a real DataChannel");

  console.log("\n== 7. communication micro-benchmarks ==");
  const micro = runMicro({ sizes: [8, 128], chunkLimits: [16384, 65536], warmup: 2, iterations: 5 });
  ok(micro.length === 2 && micro[0].tensorBytes === 8 * 384 * 4 && micro[0].packetBytes === micro[0].tensorBytes + P.HEADER_SIZE, "tensor bytes = T*384*4 and packet = tensor + header");
  ok(micro[1].chunking[0].frames === Math.ceil(micro[1].packetBytes / (16384 - P.CHUNK_HEADER_SIZE)) && micro[1].chunking[1].frames === Math.ceil(micro[1].packetBytes / (65536 - P.CHUNK_HEADER_SIZE)), "frame counts match ceil(packet / (limit - 20-byte chunk header))");
  ok(Object.values(micro[1].ops).every((o) => o.n === 5 && o.median > 0), "every micro operation produced positive timings");

  console.log("\n== 8. numerical correctness gate (this takes ~10 s) ==");
  const num = await numericalCheck(model);
  ok(num.passed, "gate passed: JS vs PyTorch within tolerance, argmax equal, distributed == local bit-for-bit, serialization bit-exact");
  ok(num.maxAbsPyTorchActivation <= 1.2e-6 && num.maxAbsPyTorchLogits <= 1.2e-6, `actual error unchanged by the optimization: activations ${num.maxAbsPyTorchActivation.toExponential(2)}, logits ${num.maxAbsPyTorchLogits.toExponential(2)} (baseline 1.07e-6 / 1.13e-6)`);
  ok(num.logitsDigestAdler32 === 1194187972, "logits digest equals the digest recorded with the ORIGINAL engine (1194187972): the optimized engine's logits are bit-identical");

  console.log("\n== 9. report generation (quick profile, real run) ==");
  const outDir = path.join(B, "results"); const label = "_unit_test";
  execFileSync("node", [B + "/run.js", "--profile", "quick", "--label", label], { stdio: "pipe" });
  const files = ["json", "md", "csv"].map((x) => path.join(outDir, `${label}.quick.${x}`));
  ok(files.every((f) => fs.existsSync(f)), "run.js writes JSON, Markdown and CSV");
  const js = JSON.parse(fs.readFileSync(files[0], "utf-8"));
  ok(js.numerics.passed && js.comparisons.length === 3 && js.comparisons.every((c) => c.failures.local.length === 0 && c.failures.distributed.length === 0), "quick run: numerics pass, 3 workloads, no failures");
  ok(js.comparisons.every((c) => /no verdict/.test(c.comparison.verdict)), "with only 3 samples per side, no local-vs-distributed verdict is issued");
  ok(/Evidence level/.test(fs.readFileSync(files[1], "utf-8")) && fs.readFileSync(files[2], "utf-8").split("\n")[0].startsWith("workload,mode"), "Markdown states the evidence level; CSV has the expected header");
  files.forEach((f) => fs.unlinkSync(f));

  console.log("\n" + "=".repeat(70));
  console.log(`STAGE 7 TESTS ${failed ? "FAILED" : "PASSED"}: ${passed} passed, ${failed} failed`);
  console.log("These are unit tests and Node simulations of the benchmark tooling. They are NOT browser, two-tab or physical-device evidence.");
  console.log("=".repeat(70));
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(2); });
