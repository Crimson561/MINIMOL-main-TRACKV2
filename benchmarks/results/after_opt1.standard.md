# MINIMOL benchmark report: after_opt1 (standard)

**Evidence level:** node-simulation-loopback (single machine, no network, no browser, no second physical device)

> Everything below was measured by running `benchmarks/run.js` on the machine described here. Local and distributed runs share one CPU, so this isolates software cost; it says nothing about real network latency.

## Environment

- Captured: 2026-11-17T09:06:06.497Z
- OS: linux 6.18.44-fc-v553 (x64)
- CPU: Intel(R) Xeon(R) Processor @ 2.10GHz (1 logical core(s), reported 0 MHz)
- Memory: 3992 MB
- Node v22.23.4 (V8 12.4.254.21-node.57)
- Git commit: 8c174c9 (working tree has uncommitted changes)

## Model and weights

- GPT-style causal decoder: 6 layers, d_model 384, 6 heads, d_ff 1536, vocab 256, context 512, 11,040,768 parameters, float32
- Seed 42; weights are randomly initialised, NOT trained (timing is unaffected, generated text is meaningless)
- Partition A 22,473,216 B (adler32 3239102982), partition B 21,689,856 B (adler32 4085685741), full 44,163,072 B
- Process memory after loading weights: RSS 177.2 MB, heap 4.3 MB, ArrayBuffers 126.4 MB; peak RSS of the whole run 222.6 MB

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
| fwd8 (T=8) | 15/15 | 0/0 | 71.63 | 66.12 | -5.51 | -7.7% | [-21.43, 7.33] | no measurable difference (95% CI of the median difference contains 0) |
| fwd35 (T=35) | 15/15 | 0/0 | 374.76 | 355.21 | -19.55 | -5.2% | [-111.64, 27.44] | no measurable difference (95% CI of the median difference contains 0) |
| fwd128 (T=128) | 9/9 | 0/0 | 1419.60 | 1522.51 | 102.92 | +7.2% | [-204.15, 387.29] | no measurable difference (95% CI of the median difference contains 0) |
| fwd512 (T=512) | 3/3 | 0/0 | 6583.04 | 7199.09 | 616.04 | +9.4% | [-287.58, 1942.41] | no verdict: only 3 samples per side (need >= 8) |
| gen (4+10 tokens) | 12/12 | 0/0 | 1135.40 | 1255.90 | 120.50 | +10.6% | [-196.80, 323.85] | no measurable difference (95% CI of the median difference contains 0) |

Generation throughput (median): local 8.846 tokens/s, distributed 7.973 tokens/s (-9.9%).

### Spread of the measurements (total ms)

| workload | mode | n | mean | median | p95 | min | max | stddev | cv |
|---|---|---|---|---|---|---|---|---|---|
| fwd8 | local | 15 | 76.30 | 71.63 | 99.71 | 58.38 | 99.87 | 14.48 | 0.190 |
| fwd8 | distributed | 15 | 77.95 | 66.12 | 124.54 | 59.31 | 145.64 | 25.15 | 0.323 |
| fwd35 | local | 15 | 388.43 | 374.76 | 523.79 | 271.95 | 523.93 | 82.79 | 0.213 |
| fwd35 | distributed | 15 | 363.76 | 355.21 | 477.25 | 285.56 | 489.32 | 66.58 | 0.183 |
| fwd128 | local | 9 | 1389.17 | 1419.60 | 1599.24 | 1008.13 | 1618.20 | 198.87 | 0.143 |
| fwd128 | distributed | 9 | 1520.97 | 1522.51 | 1928.25 | 1060.68 | 2009.15 | 295.02 | 0.194 |
| fwd512 | local | 3 | 6457.18 | 6583.04 | 6867.21 | 5889.71 | 6898.78 | 516.18 | 0.080 |
| fwd512 | distributed | 3 | 7214.14 | 7199.09 | 7768.82 | 6611.20 | 7832.12 | 610.60 | 0.085 |
| gen | local | 12 | 1151.57 | 1135.40 | 1428.98 | 875.39 | 1438.91 | 180.48 | 0.157 |
| gen | distributed | 12 | 1234.32 | 1255.90 | 1577.07 | 913.16 | 1615.66 | 234.04 | 0.190 |

CPU utilisation (process CPU time / wall time; 1.0 = one core fully busy): fwd8 local 1.00 / dist 1.00; fwd35 local 0.99 / dist 0.99; fwd128 local 0.99 / dist 0.99; fwd512 local 0.99 / dist 0.99; gen local 0.99 / dist 0.99

## Where distributed time goes (median ms per request; forward workloads)

| workload | total | A compute | B compute | serialize (A+B reply) | chunk | reassemble | decode (checksum) | validate | send calls | unattributed | protocol total (non-compute) | % non-compute |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fwd8 | 66.12 | 32.10 | 34.04 | 0.119 | 0.012 | 0.000 | 0.102 | 0.005 | 0.072 | 0.246 | 0.595 | 0.90% |
| fwd35 | 355.21 | 161.06 | 167.79 | 0.307 | 0.012 | 0.000 | 0.268 | 0.017 | 0.077 | 0.248 | 0.924 | 0.26% |
| fwd128 | 1522.51 | 826.18 | 731.06 | 0.919 | 0.015 | 0.000 | 0.872 | 0.088 | 0.101 | 0.314 | 2.297 | 0.15% |
| fwd512 | 7199.09 | 3458.37 | 3796.61 | 3.445 | 0.123 | 0.158 | 3.331 | 0.222 | 0.229 | 0.524 | 7.965 | 0.11% |

Activation packet sizes and frames per request: fwd8: 12,336 B in 1 frame(s); fwd35: 53,808 B in 1 frame(s); fwd128: 196,656 B in 1 frame(s); fwd512: 786,480 B in 4 frame(s)

Cross-check: the wrapper-measured compute times agree with the pipeline's own self-reported computeMsA/computeMsB (median A fwd8: 32.10 vs 32.10; fwd35: 161.06 vs 161.07; fwd128: 826.18 vs 826.19; fwd512: 3458.37 vs 3458.38).

## Communication micro-benchmarks (median ms, loopback, no network)

| T | tensor B | packet B | memcpy | encode | adler32 | decode | validate | B input copy | frames @64 KiB | split | reassemble | frames @256 KiB | split | reassemble |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1536 | 1584 | 0.0004 | 0.047 | 0.007 | 0.008 | 0.001 | 0.001 | 1 | 0.000 | - | 1 | 0.000 | - |
| 8 | 12288 | 12336 | 0.0004 | 0.056 | 0.052 | 0.053 | 0.004 | 0.002 | 1 | 0.000 | - | 1 | 0.000 | - |
| 35 | 53760 | 53808 | 0.0016 | 0.231 | 0.231 | 0.230 | 0.025 | 0.006 | 1 | 0.000 | - | 1 | 0.000 | - |
| 128 | 196608 | 196656 | 0.0056 | 0.846 | 0.803 | 0.821 | 0.091 | 0.021 | 4 | 0.032 | 0.041 | 1 | 0.000 | - |
| 512 | 786432 | 786480 | 0.0271 | 3.514 | 3.238 | 4.308 | 0.584 | 0.077 | 13 | 0.112 | 0.113 | 4 | 0.076 | 0.072 |

Throughput from medians (MB/s, 1 MB = 1e6 B): T=1: memcpy 4062, encode 34, adler32 234, decode 199; T=8: memcpy 30995, encode 220, adler32 238, decode 232; T=35: memcpy 34110, encode 233, adler32 233, decode 234; T=128: memcpy 35411, encode 232, adler32 245, decode 240; T=512: memcpy 29069, encode 224, adler32 243, decode 183

## Failures

None. Every attempted measured iteration completed and produced the expected tokens.

_Total benchmark wall-clock: 153.1s._
