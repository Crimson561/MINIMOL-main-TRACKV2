# Stage 7 — Performance benchmarking and optimization: status

**Status: implemented and verified at the Node-simulation and single-machine-browser level. NOT run on two physical devices or two hand-driven tabs.**
All numbers are in `BENCHMARKS.md` and `benchmarks/results/`. Measured results and analysis are labelled separately there.

## 1. Audit (before any change)
- Model: byte-level GPT decoder, 6 layers, d_model 384, 6 heads, d_ff 1536, vocab 256, context 512, 11,040,768 parameters, float32, weights random (seed 42).
- `runLive()` (Frontend/main.js) calls `PipelineA.infer(tokens)` once per generated token. Each call is a full forward pass over the whole sequence: **no KV cache**.
- One request: `MeshPartitionA.forward` → `P.encodeActivation` → `sendPacket`/`P.splitIntoChunks` → DataChannel → B `handleMessage` (`ChunkReassembler.push`, `P.decodePacket`, `P.validateActivation`) → `MeshPartitionB.forward` → `P.encodeToken` → A `_onResult`.
- Device A: embeddings + layers 0–2, 5,618,304 parameters (22,473,216 B). Device B: layers 3–5 + final norm + lm_head (last token only), 5,422,464 parameters (21,689,856 B). About equal work, run sequentially.
- Activation tensor `[1,T,384]` float32 = `1536*T` bytes; packet = 48-byte header + payload (Adler-32 over the payload); chunking only above the negotiated `maxMessageSize` (20 bytes per frame).
- Existing instrumentation gaps (serialization not isolated, B's decode time not recorded) were handled by non-invasive wrappers in the benchmark code instead of editing the pipeline.

## 2. What was built (all new files unless stated)
- `benchmarks/lib/` stats, environment capture, model loader, instrumentation (wrapper-based phase timing), loopback link without artificial delay, local/distributed runners, communication micro-benchmarks, numerical gate.
- `benchmarks/run.js` (profiles `quick`, `standard`) + `report.js` → JSON, Markdown, CSV in `benchmarks/results/`.
- `benchmarks/optimize_compare.js` (old engine vs new engine, interleaved), `matmul_variants.js`, `attention_variants.js`, `profile_workload.js`, `profile_long.js`, `analyze_profile.js`, `investigate_gen_gap.js`.
- `benchmarks/browser/bench.html` + `benchmarks/bench_browser.js`: one Chromium page, two `RTCPeerConnection`s, real `RTCDataChannel`.
- `benchmarks/baseline/` copies of the engine before optimization (`.baseline.js`, the git HEAD version) and after optimization 1 (`.opt1.js`).
- `tests/test_stage7_benchmarks.js` (46 checks).
- **Modified:** `engine/inference_engine.js`, only `matmulTransposeB` (CRLF preserved). `benchmarks/lib/model.js` takes an optional engine module. Docs: `BENCHMARKS.md`, `STAGE7_STATUS.md`, `TESTING.md` and `ARCHITECTURE.md` (Stage 7 sections appended).
- `tests/benchmark.js` and `BENCHMARK_RESULTS.json` (the Stage 4-era benchmark) were left untouched and not overwritten.

## 3. Results in one paragraph
Compute is 99.2–99.9% of a distributed request; serialization, chunking, reassembly, validation and the loopback/DataChannel transfer are 0.1–0.9%. Local and distributed inference showed no statistically measurable difference in 9 of 10 Node comparisons (the one exception, distributed generation 20% slower, did not reproduce and reversed sign in an interleaved re-run). Distribution cannot reduce single-request latency in a sequential A→B pipeline; its benefit is that each device holds about half the weights. One optimization (register-blocked matmul) made inference about 2.0–2.4× faster with bit-identical logits; a second (attention loop interchange) was bit-identical but not demonstrable end-to-end and was reverted.

## 4. Tests executed on the final code (2026-11-17)
| Suite | Result |
|---|---|
| `tests/test_tensor.js` | 38 passed |
| `tests/test_protocol.js` | passed |
| `tests/test_js_engine.js` | passed |
| `tests/test_distributed_inference.js` | passed |
| `tests/test_stage5_pipeline.js` | 50 of 50 |
| `tests/test_stage6_reliability.js` | 24 of 24 |
| `tests/test_stage7_benchmarks.js` | 46 of 46 |
| `python model/verify_reference.py` | all PyTorch reference checks passed |
| `tests/browser/stage6_two_page.js` (headless Chromium, two pages, one machine) | 21 of 21 |

Numerical gate (before and after the optimization): max |JS − PyTorch| activations 1.07e-6, logits 1.13e-6 (tolerance 1e-5); argmax equal on all six cases; distributed logits bit-identical to local; serialization bit-exact; logits digest 1194187972 unchanged, so the optimized engine's logits are bit-identical to the original's on those cases. Bit-for-bit equivalence is claimed only for what was tested (the fixture cases and the 8 matmul shapes), not proven for all inputs.

## 5. Known limitations and honest caveats
- Measured on one shared single-core VM with about 13% noise floor; many differences are inside noise; T=512 had n=3 in the standard runs.
- Weights are random. Local and distributed share one CPU in every run.
- No real network time is measured anywhere. Browser DataChannel numbers are same-page loopback.
- Browser timer resolution is 0.05–0.1 ms.
- Not run: two hand-driven tabs, two physical devices (instructions in `BENCHMARKS.md` section 12).
- Remaining performance headroom: KV cache (changes the engine contract; needs your approval), faster attention at long T, running compute off the main thread (UI freeze).
- A harness bug (my copy of the transport lacked the "wait while connecting" step) caused intermittent browser setup failures. Fixed and verified (0 setup failures in the final runs). The app's own transport already had that step.

## 6. Learning summary
**Concepts covered:** latency vs throughput; bandwidth vs latency; communication and serialization overhead; CPU-bound vs communication-bound; warm-up and JIT; median vs mean vs P95; measurement variance and noise floor; interleaved (ABBA) comparison; bootstrap confidence intervals; probe effect and timer resolution; sampling profilers and why inlining distorts them; self time vs total time; register blocking and why preserving summation order keeps results bit-identical; why local can never be beaten by a sequential two-device pipeline.
**Files understood:** `engine/inference_engine.js`, `engine/mesh_pipeline.js`, `protocol/mesh_protocol.js`, `Frontend/main.js` (transport, `runLive`), `model/export_weights.py`, `tests/test_stage5_pipeline.js`, and the new `benchmarks/`.
**Performance issues identified:** matmul dominates; attention grows with T²; no KV cache; load-time checksum 332 ms; main-thread compute.
**Optimizations:** (1) register-blocked matmul, kept, about 2×; (2) attention loop interchange, reverted for lack of end-to-end evidence; 4×4 tile rejected.
**You should be able to explain independently:**
1. Why the median, not the mean, and why warm-up runs are discarded.
2. Why the mock `setTimeout(1)` link is not network time.
3. Why a two-device pipeline cannot reduce one request's latency, and what it does buy.
4. Why `matmul` blocking changed nothing numerically but reordering the sum would.
5. Why a 26% speedup of a kernel that is 15% of the run is invisible on a noisy machine.

**Reasoning questions (answers are in the measurements):**
1. If device B were 3× faster than A, would distributed beat local? (Hint: sum the sequential times, and note the larger transfer cost.)
2. Why did the first 3-sample quick run claim "distributed 44% faster"? Which two methodology rules did it violate?
3. The checksum is 130× slower than memcpy yet we did not optimize it. Under what change of conditions would it become worth optimizing?

## 7. Reproduce
See `BENCHMARKS.md` section 11.
