# MINIMOL — Stage 7 Benchmarks

Everything in this file was measured by the scripts in `benchmarks/`; the raw JSON/CSV files are in `benchmarks/results/`. Anything that is an estimate or a deduction rather than a measurement is labelled **(analysis)**.

## 0. Read this first: what these numbers are and are not

| Evidence level | Done? | What it tells you |
|---|---|---|
| Node simulation, one process, loopback link (`benchmarks/run.js`) | **Yes** | Software cost of the pipeline (compute, serialization, chunking, checksums). No network. |
| Headless Chromium, one page, two `RTCPeerConnection`s, real `RTCDataChannel` (`benchmarks/bench_browser.js`) | **Yes** | Real DataChannel behaviour on loopback ICE, browser JIT. Same machine, no physical network. |
| Two browser tabs through the room-server | **No** | Not run in Stage 7. |
| Two physical devices | **No** | Not run. Cannot be produced in the sandbox. Nothing here may be quoted as a physical-device result. |

The Stage 5 test `Link` uses a fixed 1 ms `setTimeout`. That is fake latency and is **never** used for any number below.

## 1. Environment (as reported by the runtime)

- Sandbox VM: Intel Xeon @ 2.10 GHz, **1 logical core**, 3,992 MB RAM, Linux 6.18 x64. A shared virtual machine; the CPU frequency is not reported.
- Node v22.23.4 (V8 12.4). Browser: headless Chromium, `Chrome/153.0.8010.12`. PyTorch 2.15.0 (CPU use only for weight export / reference).
- Git commit `8c174c9`. The tree is reported "dirty" because the supplied zip has CRLF line endings against an LF git index (pre-existing) plus the Stage 7 changes listed in `docs/history/STAGE7_STATUS.md`.
- **Noise floor of this machine:** an allocation-free arithmetic loop repeated 40 times had a coefficient of variation (CV) of 0.13 (min 2.94 ms, max 5.44 ms). Differences smaller than roughly 15% between two single runs are not meaningful here. This is why comparisons use interleaving and a bootstrap confidence interval (section 2).

## 2. Methodology

- **Model:** byte-level GPT decoder, 6 layers, d_model 384, 6 heads, d_ff 1536, vocab 256, context 512, 11,040,768 float32 parameters, seed 42. Weights are random (not trained): timing is unaffected, generated text is meaningless. Partition A 22,473,216 B (adler32 3239102982), partition B 21,689,856 B (adler32 4085685741), full model 44,163,072 B. These checksums equal the ones committed in `tests/fixtures/stage5_reference.json`, and the runner refuses to start if the weights do not verify.
- **Inputs (fixed):** forward workloads use token `i` = `(37*i + 11) mod 256` for T = 8, 35, 128, 512. Generation uses the prompt `MESH` plus 10 greedy tokens (14 tokens total). Every generation is compared with the PyTorch reference sequence; every forward result is compared with the expected next token. A wrong answer is a **failure**, not a data point.
- **Warm-up:** 1–3 discarded iterations first (V8 compiles hot loops after they have run). A failing warm-up aborts the run.
- **Interleaving (ABBA):** local and distributed runs alternate in rounds, and the order flips every round, so machine drift and JIT state hit both equally. The first quick run showed why: with one round, whichever mode ran second looked 44% faster at T=8, which is impossible for a sequential pipeline.
- **Statistics:** n, mean, median, P95 (linear interpolation), min, max, sample standard deviation, CV. With fewer than 20 samples the P95 is essentially the second-largest sample. The headline is the **median**. The local-vs-distributed verdict uses a seeded bootstrap 95% confidence interval of `median(distributed) - median(local)`; if the interval contains 0 the report says "no measurable difference". Fewer than 8 samples per side gives **no verdict**.
- **Failures:** recorded in a separate error list with the error code; never mixed into latency statistics.
- **No double counting:** the Node instrumentation wraps synchronous calls that never contain each other, so spans are disjoint; `unattributed = total - sum(spans)` is reported, not hidden. The wrappers' compute times agree with the pipeline's own `computeMsA/computeMsB`. The browser timeline is defined by consecutive timestamps on one clock, so its residual is 0 *by construction* (it is not an independent check).
- **Probe effect:** each wrapper adds two `performance.now()` calls. The measured protocol spans are 0.005–3.4 ms, so the effect is small but not zero.
- **Timer resolution:** browsers clamp `performance.now()` (observed 0.05–0.1 ms steps). Browser timings below about 1 ms are coarse, and the 1 KB browser transfer (0–0.3 ms) is **below the timer's resolution**; its throughput figure is meaningless.

## 3. Local vs distributed (Node, loopback, ABBA) — original engine

`benchmarks/results/baseline.standard.*` (3 rounds, 219 s). Medians in ms. "Distributed" = full PipelineA → chunk → loopback → PipelineB → reply on the same CPU.

| workload | n (L/D) | local median | distributed median | abs diff | % diff | 95% CI of diff | verdict |
|---|---|---|---|---|---|---|---|
| T=8 | 15/15 | 125.3 | 136.9 | +11.6 | +9.3% | [-10.9, 35.9] | no measurable difference |
| T=35 | 15/15 | 545.5 | 531.4 | -14.1 | -2.6% | [-44.2, 24.7] | no measurable difference |
| T=128 | 9/9 | 1973.5 | 2055.0 | +81.5 | +4.1% | [-11.2, 259.4] | no measurable difference |
| T=512 | 3/3 | 9629.6 | 9584.6 | -45.0 | -0.5% | — | no verdict (n=3) |
| generate 4+10 | 12/12 | 1286.3 | 1541.9 | +255.6 | +19.9% | [33.6, 439.0] | distributed slower |

Generation throughput (median): local 7.77 tokens/s, distributed 6.51 tokens/s.

**The one "significant" result did not hold up.** An ABBA re-check (8 pairs) gave 1304 vs 1416 ms (+8.6%), and the later fully interleaved before/after run measured distributed generation *faster* than local (1453 vs 1659 ms) on the old engine. A real structural cost cannot change sign, and the measured protocol cost is only about 6 ms of the 1542. I attribute the earlier gap to machine drift/ordering, but I did not isolate its cause beyond that. With 10 comparisons at 95%, one false positive is not surprising.

## 4. Local vs distributed — after optimization 1 (matmul blocking)

`benchmarks/results/after_opt1.standard.*` (153 s). Medians in ms.

| workload | n (L/D) | local | distributed | abs diff | % diff | 95% CI of diff | verdict |
|---|---|---|---|---|---|---|---|
| T=8 | 15/15 | 71.6 | 66.1 | -5.5 | -7.7% | [-21.4, 7.3] | no measurable difference |
| T=35 | 15/15 | 374.8 | 355.2 | -19.6 | -5.2% | [-111.6, 27.4] | no measurable difference |
| T=128 | 9/9 | 1419.6 | 1522.5 | +102.9 | +7.2% | [-204.1, 387.3] | no measurable difference |
| T=512 | 3/3 | 6583.0 | 7199.1 | +616.0 | +9.4% | — | no verdict (n=3) |
| generate 4+10 | 12/12 | 1135.4 | 1255.9 | +120.5 | +10.6% | [-196.8, 323.9] | no measurable difference |

Generation throughput: local 8.85 tokens/s, distributed 7.97 tokens/s (not significant).

### Does distribution help?

**No, not for latency, and it cannot here.** (analysis) A pipeline split runs A, then B, never in parallel for one request, so one request costs `A + serialize + transfer + B + reply`, which is at least the local cost `A + B`. The measurements agree: no comparison shows distributed significantly faster, and several medians are slightly higher. What the split does provide is **memory**: each device holds one partition (21.43 MB and 20.69 MB of weights) instead of 42.12 MB. That number comes from the partition file sizes; I did not measure per-device process memory on separate devices. Both halves also ran on one core here, so no hardware-parallelism benefit could appear even in principle.

## 5. Where distributed time goes (Node, after optimization 1; medians, ms)

| T | total ≈ | A compute | B compute | serialize | chunk | reassemble | decode (incl. checksum) | validate | send calls | unattributed | packet bytes | frames |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 66.1 | 32.1 | 34.0 | 0.119 | 0.012 | 0.000 | 0.102 | 0.005 | 0.072 | 0.246 | 12,336 | 1 |
| 35 | 355.2 | 161.1 | 167.8 | 0.307 | 0.012 | 0.000 | 0.268 | 0.017 | 0.077 | 0.248 | 53,808 | 1 |
| 128 | 1522.5 | 826.2 | 731.1 | 0.919 | 0.015 | 0.000 | 0.872 | 0.088 | 0.101 | 0.314 | 196,656 | 1 |
| 512 | 7199.1 | 3458.4 | 3796.6 | 3.445 | 0.123 | 0.158 | 3.331 | 0.222 | 0.229 | 0.524 | 786,480 | 4 |

Non-compute ("protocol") time is 0.10–0.90% of a request (0.6 ms at T=8, 8.0 ms at T=512). The process is CPU-bound: process CPU time / wall time was 0.99–1.00 in every workload. Activation size is exact: `48 + 1536*T` bytes; chunking adds 20 bytes per frame and happens only above the channel limit (262,144 B here).

## 6. Communication micro-benchmarks (Node, loopback, medians, ms)

Each step timed alone on identical activation bytes; `memcpy` is the lower bound for anything that touches every byte.

| T | tensor B | packet B | memcpy | encode | adler32 | decode | validate | B input copy | split @64 KiB (frames) | reassemble @64 KiB | split @256 KiB (frames) | reassemble @256 KiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 12,288 | 12,336 | 0.0004 | 0.056 | 0.052 | 0.053 | 0.004 | 0.002 | 0.000 (1) | — | 0.000 (1) | — |
| 35 | 53,760 | 53,808 | 0.0016 | 0.231 | 0.231 | 0.230 | 0.025 | 0.006 | 0.000 (1) | — | 0.000 (1) | — |
| 128 | 196,608 | 196,656 | 0.0056 | 0.846 | 0.803 | 0.821 | 0.091 | 0.021 | 0.032 (4) | 0.041 | 0.000 (1) | — |
| 512 | 786,432 | 786,480 | 0.0271 | 3.514 | 3.238 | 4.308 | 0.584 | 0.077 | 0.112 (13) | 0.113 | 0.076 (4) | 0.072 |

Throughput from medians: memcpy ≈ 29,000–35,000 MB/s; encode/decode/adler32 ≈ 183–245 MB/s. The checksum (`adler32`, one `%` per byte) is about 90–95% of encode and decode, and about **130× slower than a memory copy**. It is still tiny next to compute (0.2 ms vs 160 ms at T=35), so it is not a request-time bottleneck. One-time cost at startup: verifying the 44.2 MB of weights took 332 ms (measured by wrapping `adler32` during `loadModel`).

## 7. Browser results (headless Chromium, one page, real RTCDataChannel on loopback ICE)

`benchmarks/results/browser_current.browser.json` (optimized engine), 10 measured iterations per workload, 0 failures, 0 setup failures. Negotiated `maxMessageSize` = 262,144 B (what `pc.sctp` reported). Ping-pong RTT of a 1-byte message: median 0.20 ms, P95 0.30 ms (60 samples, 50 kept; **0.1 ms timer resolution**, so treat as an order of magnitude).

One-way bulk transfer, first `send` call to last frame handled at the other peer (same clock), 15 samples each:

| bytes | frames | median ms | P95 ms | median MB/s | best MB/s |
|---|---|---|---|---|---|
| 1,024 | 1 | 0.10 | 0.30 | below timer resolution | — |
| 12,336 (T=8 activation) | 1 | 0.80 | 0.93 | 15.4 | 17.6 |
| 53,808 (T=35 activation) | 1 | 2.80 | 3.36 | 19.2 | 19.9 |
| 196,656 (T=128 activation) | 1 | 9.70 | 10.65 | 20.3 | 21.9 |
| 786,480 (T=512) | 4 | 31.50 | 52.53 | 25.0 | 35.0 |
| 4,194,304 | 17 | 191.80 | 211.88 | 21.9 | 26.7 |

The earlier baseline-engine browser run measured similar rates (about 21–29 MB/s). These are loopback rates through the browser's software SCTP/DTLS stack on one shared core, **not** the bandwidth of any real network, and the run-to-run spread of about ±25% shows the machine's noise. A real Wi-Fi/Internet link will differ, probably by a lot.

Browser local vs distributed (medians, ms; n = 10 each; all "no measurable difference"):

| workload | local | distributed | % diff | 95% CI of diff |
|---|---|---|---|---|
| T=8 | 104.0 | 89.9 | -13.5% | [-26.4, 37.2] |
| T=35 | 338.3 | 402.2 | +18.9% | [-127.4, 191.1] |
| T=128 | 1459.0 | 1434.2 | -1.7% | [-401.0, 291.1] |
| generate 4+10 | 1085.7 | 1082.5 | -0.3% | [-327.4, 424.0] |

CV in the browser was 0.13–0.28, so the sign of these small differences is noise. Browser phase breakdown (medians, ms; timestamps from one clock; 0.05 ms steps): at T=35, A compute 241.3, serialize 0.3, A→B wire 3.25, B non-compute processing 0.55, B compute 182.6, B→A wire 0.65. Over a whole 10-token generation, A→B wire ≈ 14.3 ms and B→A wire ≈ 5.8 ms of ≈ 1083 ms (about 1.9%). In the browser the real DataChannel wire time is 3.25 ms for a 54 KB activation, versus 0.08 ms for the loopback copy in Node; communication is larger than in the Node simulation but still about 1% of a request.

Browser before/after (original engine in `browser_baseline.browser.json`, distributed medians): T=8 244→90 ms, T=35 650→402, T=128 3026→1434, generate 1548→1083. **These two browser runs were separate processes at different times, not interleaved**, so treat them as supporting evidence only; the Node interleaved comparison in section 8 is the primary result.

## 8. Optimizations

Profile first (Node `--cpu-prof`, `--no-turbo-inlining` so V8 does not merge `matmulTransposeB` into its callers; `benchmarks/profile_workload.js`, `benchmarks/analyze_profile.js`). Short sequences (distributed generation, T = 4–13): `matmulTransposeB` **89.2%** of self time, `adler32` 5.5% (almost all of it the 332 ms load-time weight verification), `gelu` 1.5%, attention 0.3%. Without `--no-turbo-inlining` the same profile attributes 28% to `causalSelfAttention` and `forward` because the inlined matmul is charged to its callers, a reminder that profiler output needs interpretation.

### Optimization 1 — register-blocked `matmulTransposeB` (KEPT)

Bottleneck: the naive loop recomputes one output at a time and reloads `A[k]` for every output column. Change: compute a 2-row × 4-column tile per pass (8 accumulators), so each loaded `A[k]` serves 4 outputs and each `B[k]` serves 2. Each output still starts from its bias and adds products for `k = 0..K-1` in order, in double precision, so the result is **bit-identical** (verified on 8 shapes with and without bias, and by the logits digest).

Before/after: original engine and new engine loaded in the same process, same weights, ABBA, 4 rounds (`benchmarks/results/opt1_matmul_blocking.before_after.json`). Medians in ms:

| workload | local before→after | speedup | distributed before→after | speedup | all CIs |
|---|---|---|---|---|---|
| T=8 (n=12) | 148.0 → 62.7 | 2.36× | 159.5 → 65.9 | 2.42× | exclude 0 |
| T=35 (n=12) | 623.2 → 308.3 | 2.02× | 606.9 → 276.6 | 2.19× | exclude 0 |
| T=128 (n=8) | 2333.4 → 1018.4 | 2.29× | 2410.4 → 1124.4 | 2.14× | exclude 0 |
| generate (n=8) | 1658.8 → 761.5 | 2.18× | 1453.0 → 726.1 | 2.00× | exclude 0 |

Generation throughput: local 6.06 → 13.13 tokens/s, distributed 6.88 → 13.78 tokens/s. No failures. At T=512 the standard runs show 9630 → 6583 ms local (about 1.46×; those runs were not paired and used n=3), which is smaller; the T=512 profile after the change is 68.9% matmul and 24.9% `causalSelfAttention`, because attention grows as O(T²).

Correctness after the change: logits digest unchanged (1194187972), max |JS−PyTorch| unchanged (activations 1.07e-6, logits 1.13e-6, tolerance 1e-5), distributed == local bit-for-bit, all Stage 1–6 tests pass.

### Optimization 2 — attention weighted-sum loop interchange (REVERTED)

Idea: make `j` the outer loop so V rows are read contiguously (the old loop jumped 1,152 floats between reads). Isolated kernel result, bit-identical: T=35 0.77→0.55 ms (-29%), T=128 10.65→6.77 ms (-36%), T=512 158.5→117.3 ms (-26%), significant. End-to-end against the opt-1 engine (interleaved, `opt2_attention_interchange.before_after.json`): T=128 local -23.7% (CI [-543.9, 8.9]), T=512 local -4.2% (CI [-806.0, 1321.1]), distributed -9.5% and -10.4%, all **within noise**. My rule was to keep a change only if the end-to-end gain is demonstrable, so I reverted it. (analysis) The kernel accounts for roughly 15% of a T=512 forward, so a 26% gain there predicts about 4%, which matches the −4.2% observed and is below what this machine can resolve.

### Candidates measured and rejected, or not touched

- 4×4 matmul tile: no gain over 2×4 (10.17 vs 10.03 ms at T=35; about 8% at T=128, within noise). Rejected.
- `adler32`: 130× slower than memcpy but 0.1–0.9% of a request, plus 332 ms once at startup. Not justified. Left unchanged.
- B's defensive `new Float32Array(activations)` copy: 0.002–0.077 ms. Not touched.
- GC: about 0.4% of profile time. Not touched.
- Not done by design: KV cache (changes the engine contract, needs approval), reordering summation (would no longer be bit-identical), WebAssembly/SIMD/GPU/threads (outside the Stage 7 boundaries).

## 9. Identified bottlenecks (measured)

1. **Model compute dominates**: 99.2–99.9% of every distributed request (by median, after opt 1; the per-request protocol share of 0.10–0.90% is the more reliable figure, because adding the independent medians of A and B can exceed 100% on this noisy machine). After opt 1, `matmulTransposeB` is still 80% of CPU at short T and 69% at T=512.
2. **Attention is the second cost at long T** (24.9% at T=512), growing as O(T²).
3. **No KV cache** (code reading, not measured): every generated token recomputes the whole sequence, so the compute for generating N tokens grows with the sum of sequence lengths. Adding a cache would be the largest remaining win but changes the engine contract.
4. **Not bottlenecks (measured):** serialization, chunking, reassembly, validation, and the DataChannel on loopback.
5. **Not measured at all:** real network time between two devices, DataChannel behaviour under loss or real bandwidth, the UI freezing during main-thread compute (Stage 5 limit), mobile CPUs.

## 10. Known limitations

- One shared core, noisy VM; many comparisons are inside noise. T=512 had only 3 samples per side in the standard runs (no verdict issued).
- Random (untrained) weights.
- Local and distributed share one CPU in every run here.
- Browser timer resolution 0.05–0.1 ms.
- The browser harness once failed to establish the pipeline (HELLO not received, with both channels open). Cause found: my harness transport lacked the app's "wait if the channel is still connecting" step. After fixing it, both browser runs had 0 setup failures (the earlier flakes are recorded in the history above and are not in the final results). The app's own `makeTransport` already contains that wait.
- `benchmarks/results/baseline.standard.json` predates the raw-sample field (`rawTotalMs`) added afterwards; later result files include raw samples.
- Physical two-device and two-tab benchmarks were not run.

## 11. Reproduce

```
python model/export_weights.py && python model/export_stage5_reference.py   # weights + PyTorch reference (needs torch)
node benchmarks/run.js --profile standard --label mine                      # local vs distributed, micro, numerics -> benchmarks/results/mine.standard.{json,md,csv}
node benchmarks/run.js --profile quick --label smoke                        # ~20 s smoke run (no verdict, n=3)
node benchmarks/optimize_compare.js mylabel 4 inference_engine.baseline.js  # before/after, interleaved
node benchmarks/bench_browser.js mine current 10                            # needs Chromium: CHROME_PATH and puppeteer-core
node --cpu-prof --cpu-prof-dir=benchmarks/results/profiles --no-turbo-inlining benchmarks/profile_workload.js 3
node benchmarks/analyze_profile.js benchmarks/results/profiles/<file>.cpuprofile
```

`bench_browser.js` needs `npm install puppeteer-core` (not a project dependency) and `CHROME_PATH=/path/to/chrome` (optionally `PUPPETEER_CORE_PATH`); it prints a message and exits if either is missing. Profiler output (`benchmarks/results/profiles/`) is not committed, because it embeds local file paths; the two profiling commands above regenerate it. Expect different absolute numbers on different hardware; compare ratios, not milliseconds.

## 12. For the physical-device run you still need to do

Open the Frontend on two devices, join one room, run an inference, and record from the app's per-request metrics (`computeMsA`, `computeMsB`, `rttMs`, `totalMs`, activation bytes). An approximation of network + B-side non-compute time is `rttMs - computeMsB` (analysis, not a direct measurement). Record device models, OS/browser versions, and the network type. Report those numbers in a separate section, never merged with the ones above.

## 13. Older measurement: `BENCHMARK_RESULTS.json` (Stage 4, not comparable to the sections above)

The repository root also contains `BENCHMARK_RESULTS.json`, written by `tests/benchmark.js` on **2026-11-01 on a Windows x64 machine with Node v24.18.0**. It is the only measurement in the project made on the author's own hardware, but it is a different experiment from everything above:

- It is a single run (no warm-up, no repetitions, no confidence interval) of 20 generated bytes, with a longer prompt, so sequences reach about 54 bytes. With no KV cache every step recomputes the whole sequence, so tokens/s depends strongly on sequence length. It recorded **2.98 tokens/s** and 1,338,738 wire bytes, using a simulated A to B hand-off in one process (no WebRTC, no network).
- Stage 7's headline generation throughput (6.5–13.8 tokens/s depending on engine version and mode) used a 4-byte prompt plus 10 tokens (sequences of at most 14), on a different machine and a newer engine. **Do not compare the two numbers.**
- Stage 8 re-ran `tests/benchmark.js` on a scratch copy on the 1-core Linux VM (current engine): 3.07 tokens/s, 163 / 161 ms mean compute per partition, 0.64 / 0.52 ms mean serialization / deserialization, identical wire bytes (1,338,738). That run was not saved to the repository. Running the script yourself **overwrites** `BENCHMARK_RESULTS.json`.

## 14. Stage 8 re-check

The Stage 8 audit re-ran the quick profile (`node benchmarks/run.js --profile quick`) on a scratch copy: numerical gate passed (maximum difference from PyTorch 9.7e-7 for activations and 8.9e-7 for logits, tolerance 1e-5), 0 failures, protocol cost 1.0 ms of ~62 ms at T=8. The quick profile has only 2–3 samples per side and issues no verdicts; it is a smoke test, not a result. No numbers in sections 3–8 were modified in Stage 8.
