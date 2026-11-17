# MINIMOL benchmark report: baseline (standard)

**Evidence level:** node-simulation-loopback (single machine, no network, no browser, no second physical device)

> Everything below was measured by running `benchmarks/run.js` on the machine described here. Local and distributed runs share one CPU, so this isolates software cost; it says nothing about real network latency.

## Environment

- Captured: 2026-11-15T17:09:54.642Z
- OS: linux 6.18.44-fc-v470 (x64)
- CPU: Intel(R) Xeon(R) Processor @ 2.10GHz (1 logical core(s), reported 0 MHz)
- Memory: 3992 MB
- Node v22.23.4 (V8 12.4.254.21-node.57)
- Git commit: 8c174c9 (working tree has uncommitted changes)

## Model and weights

- GPT-style causal decoder: 6 layers, d_model 384, 6 heads, d_ff 1536, vocab 256, context 512, 11,040,768 parameters, float32
- Seed 42; weights are randomly initialised, NOT trained (timing is unaffected, generated text is meaningless)
- Partition A 22,473,216 B (adler32 3239102982), partition B 21,689,856 B (adler32 4085685741), full 44,163,072 B
- Process memory after loading weights: RSS 177.4 MB, heap 4.3 MB, ArrayBuffers 126.4 MB; peak RSS of the whole run 222.6 MB

## Method

- Rounds: 3, ABBA: local first in even rounds, distributed first in odd rounds
- Fixed inputs: forward workloads use token i = (37*i + 11) mod 256; generation uses prompt "MESH" and 10 greedy tokens (reference sequence checked every iteration)
- Warm-up iterations are discarded (listed per workload below); failed iterations are kept in a separate error list and never enter the statistics
- Channel limit used by the loopback link: 262144 bytes

## Numerical correctness gate

- Result: **PASS**
- Max |JS - PyTorch|: activations 1.07e-6, logits 1.13e-6 (tolerance 0.00001)
- Argmax equals PyTorch on every case: true; distributed logits bit-identical to local: true; serialization bit-exact: true
- Logits digest (Adler-32 over all fixture logits): 1194187972

## Local versus distributed (median wall time per iteration, ms)

| workload | n ok (L/D) | failed (L/D) | local median | distributed median | abs diff | % diff | 95% CI of diff | verdict |
|---|---|---|---|---|---|---|---|---|
| fwd8 (T=8) | 15/15 | 0/0 | 125.27 | 136.90 | 11.63 | +9.3% | [-10.92, 35.94] | no measurable difference (95% CI of the median difference contains 0) |
| fwd35 (T=35) | 15/15 | 0/0 | 545.45 | 531.35 | -14.10 | -2.6% | [-44.21, 24.75] | no measurable difference (95% CI of the median difference contains 0) |
| fwd128 (T=128) | 9/9 | 0/0 | 1973.49 | 2055.01 | 81.52 | +4.1% | [-11.20, 259.38] | no measurable difference (95% CI of the median difference contains 0) |
| fwd512 (T=512) | 3/3 | 0/0 | 9629.57 | 9584.60 | -44.97 | -0.5% | [-777.81, 1324.65] | no verdict: only 3 samples per side (need >= 8) |
| gen (4+10 tokens) | 12/12 | 0/0 | 1286.25 | 1541.86 | 255.60 | +19.9% | [33.59, 439.03] | distributed SLOWER (difference is outside measurement noise) |

Generation throughput (median): local 7.775 tokens/s, distributed 6.507 tokens/s (-16.3%).

### Spread of the measurements (total ms)

| workload | mode | n | mean | median | p95 | min | max | stddev | cv |
|---|---|---|---|---|---|---|---|---|---|
| fwd8 | local | 15 | 137.88 | 125.27 | 177.62 | 110.44 | 223.43 | 27.90 | 0.202 |
| fwd8 | distributed | 15 | 153.73 | 136.90 | 211.77 | 109.58 | 221.15 | 34.72 | 0.226 |
| fwd35 | local | 15 | 590.94 | 545.45 | 797.93 | 514.89 | 1031.20 | 133.05 | 0.225 |
| fwd35 | distributed | 15 | 557.36 | 531.35 | 670.43 | 445.70 | 847.28 | 87.77 | 0.157 |
| fwd128 | local | 9 | 2026.63 | 1973.49 | 2421.09 | 1706.87 | 2657.67 | 256.65 | 0.127 |
| fwd128 | distributed | 9 | 2091.38 | 2055.01 | 2276.83 | 1913.53 | 2310.79 | 130.27 | 0.062 |
| fwd512 | local | 3 | 9725.22 | 9629.57 | 9987.67 | 9518.64 | 10027.46 | 267.56 | 0.028 |
| fwd512 | distributed | 3 | 9892.51 | 9584.60 | 10717.42 | 9249.64 | 10843.29 | 840.26 | 0.085 |
| gen | local | 12 | 1329.19 | 1286.25 | 1521.75 | 1138.02 | 1542.36 | 117.71 | 0.089 |
| gen | distributed | 12 | 1540.29 | 1541.86 | 1807.78 | 1173.40 | 1833.40 | 218.20 | 0.142 |

CPU utilisation (process CPU time / wall time; 1.0 = one core fully busy): fwd8 local 0.99 / dist 0.99; fwd35 local 0.99 / dist 0.99; fwd128 local 0.99 / dist 0.99; fwd512 local 0.99 / dist 0.99; gen local 1.00 / dist 1.00

## Where distributed time goes (median ms per request; forward workloads)

| workload | total | A compute | B compute | serialize (A+B reply) | chunk | reassemble | decode (checksum) | validate | send calls | unattributed | protocol total (non-compute) | % non-compute |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fwd8 | 136.90 | 67.48 | 68.69 | 0.139 | 0.015 | 0.000 | 0.098 | 0.005 | 0.071 | 0.270 | 0.600 | 0.44% |
| fwd35 | 531.35 | 264.46 | 271.45 | 0.281 | 0.014 | 0.000 | 0.249 | 0.014 | 0.079 | 0.232 | 0.884 | 0.17% |
| fwd128 | 2055.01 | 1042.47 | 1097.52 | 0.850 | 0.016 | 0.000 | 0.772 | 0.048 | 0.104 | 0.260 | 2.087 | 0.10% |
| fwd512 | 9584.60 | 4812.82 | 4975.24 | 3.131 | 0.166 | 0.145 | 3.014 | 0.197 | 0.230 | 0.386 | 7.632 | 0.08% |

Activation packet sizes and frames per request: fwd8: 12,336 B in 1 frame(s); fwd35: 53,808 B in 1 frame(s); fwd128: 196,656 B in 1 frame(s); fwd512: 786,480 B in 4 frame(s)

Cross-check: the wrapper-measured compute times agree with the pipeline's own self-reported computeMsA/computeMsB (median A fwd8: 67.48 vs 67.49; fwd35: 264.46 vs 264.46; fwd128: 1042.47 vs 1042.48; fwd512: 4812.82 vs 4812.82).

## Communication micro-benchmarks (median ms, loopback, no network)

| T | tensor B | packet B | memcpy | encode | adler32 | decode | validate | B input copy | frames @64 KiB | split | reassemble | frames @256 KiB | split | reassemble |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1536 | 1584 | 0.0002 | 0.007 | 0.006 | 0.006 | 0.001 | 0.001 | 1 | 0.000 | - | 1 | 0.000 | - |
| 8 | 12288 | 12336 | 0.0003 | 0.046 | 0.043 | 0.044 | 0.003 | 0.002 | 1 | 0.000 | - | 1 | 0.000 | - |
| 35 | 53760 | 53808 | 0.0014 | 0.201 | 0.189 | 0.192 | 0.013 | 0.006 | 1 | 0.000 | - | 1 | 0.000 | - |
| 128 | 196608 | 196656 | 0.0045 | 0.790 | 0.706 | 0.708 | 0.046 | 0.020 | 4 | 0.026 | 0.027 | 1 | 0.000 | - |
| 512 | 786432 | 786480 | 0.0232 | 2.978 | 2.830 | 2.837 | 0.186 | 0.077 | 13 | 0.113 | 0.112 | 4 | 0.095 | 0.068 |

Throughput from medians (MB/s, 1 MB = 1e6 B): T=1: memcpy 7402, encode 229, adler32 276, decode 254; T=8: memcpy 36443, encode 266, adler32 284, decode 280; T=35: memcpy 39247, encode 267, adler32 284, decode 280; T=128: memcpy 43489, encode 249, adler32 278, decode 278; T=512: memcpy 33939, encode 264, adler32 278, decode 277

## Failures

None. Every attempted measured iteration completed and produced the expected tokens.

_Total benchmark wall-clock: 219s._
