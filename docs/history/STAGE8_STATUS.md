# MINIMOL — Stage 8 Status (documentation, portfolio preparation, showcase)

Audit and validation date: 2026-11-17. Environment: Linux x86-64, 1 core, Node 22.23.4, Python 3.12.3, torch 2.15.0, numpy 2.5.4, Playwright Chromium (headless).
**Nothing has been committed or published.** All changes are in the working tree for you to review.

> **Correction (same day, after you reported the project "still not working as a whole").** My first version of this report called the project verified. That was too strong: I had only run Node simulations and the signalling test, and I had not run the app in a browser. Running it properly found a real bug that made the app unusable as a pair of devices (see §2, rows 20–23). Sections 1, 2, 4, 9 and 12 below are updated.

## 1. Verdict in three lines

- **Works end to end in a real browser on one machine:** two headless Chromium pages create a room, join, download and verify their halves, open a WebRTC DataChannel, and generate bytes through both partitions, driven through the visible UI (13-check test) and also from a non-secure `http://<ip>` origin. **Not yet shown on two physical devices.**
- **The generated text is meaningless** (random weights). It was always so, but the UI did not say it, which made a working pipeline look broken. It does now.
- **Reproducible and documented,** with claims labelled by evidence level; the docs previously skipped the **Inference** tab, now fixed.

## 2. What the audit found and what was done

| # | Finding | Action |
|---|---|---|
| 1 | **Stage 5–7 work was never committed** (`mesh_pipeline.js`, `connection_state.js`, `benchmarks/`, stage 6/7 tests, status docs, BENCHMARKS.md were untracked). A push of HEAD would be broken | **Not committed** (you decide). Commit plan in §9 |
| 2 | **775 `node_modules` files tracked**; not ignored | `git rm -r --cached` (index only; files untouched); `node_modules/` added to `.gitignore` |
| 3 | **`.gitignore` rule `lib/` silently ignored `benchmarks/lib/`** (8 files, the core of the benchmark harness) | Anchored to `/lib/`; confirmed all 8 files are now publishable and the benchmark runs from a clean copy |
| 4 | Generated weights (~88 MB) not ignored | `room-server/public/weights/` ignored; regenerate with `python model/export_weights.py` |
| 5 | CRLF/LF mismatch made 18 files look modified for no reason | `.gitattributes` (`* text=auto eol=lf`). Run `git add --renormalize .` before the first commit |
| 6 | README: titled "MESH", claimed "40 tests" from an `npm test` that was a placeholder, never said the model is untrained, implied phones were tested, MiB shown as MB | Rewritten |
| 7 | `DEMO_GUIDE.md` described an old UI and made untested claims (iPhone Safari, "20–30 ms") | Moved to `docs/history/DEMO_GUIDE_stage4_outdated.md` with an OUTDATED banner; replaced by `DEMO_SCRIPT.md` |
| 8 | No automated signalling tests | Added `room-server/test/signalling.test.mjs` (15 checks, own server on a random port); mutation-checked (it fails when `MAX_PEERS` is wrong); `npm test` now works |
| 9 | **"Copy code" button used `navigator.clipboard`**, which does not exist over plain `http://<LAN-IP>` | Added a fallback in `Frontend/main.js`. **Syntax-checked only, not tried in a browser** |
| 10 | Server: leftover debug handler (`ping-server`, a broadcast to every client), non-English debug logs, hard-coded port, raw stack trace on port clash | Removed/English logs, `PORT` env var, friendly `EADDRINUSE` message |
| 11 | Tests crashed with a raw stack trace when weights were missing | `test_js_engine.js` and `test_distributed_inference.js` now print "Weights missing. Run: python model/export_weights.py" (the Stage 5 test already did) |
| 12 | `bench_browser.js` hard-coded sandbox paths (`/opt/npm-tools/...`, `/opt/pw-browsers/...`) | Uses `puppeteer-core` and `CHROME_PATH` env vars; friendly message if missing |
| 13 | Stale `// VERIFY` comments in `main.js` ("room-server was missing") | Removed/updated |
| 14 | Package metadata: ISC licence in `package.json` vs MIT `LICENSE`; no engines/scripts | MIT, `engines`, `start` and `test` scripts; `socket.io-client` as a dev dependency |
| 15 | No Python dependency file | `requirements.txt` with the verified versions |
| 16 | `BENCHMARK_RESULTS.json` (Windows, Stage 4) not explained | BENCHMARKS.md §13 labels it as an older, non-comparable single run. The file itself was not modified |
| 17 | Stale/unused files | Deleted (approved): root `main.html`/`main.css` (outdated duplicates, not served), `satic/main.py` (unrelated Flask scratch), `room-server/test-client.html`/`test-server.html` (no longer compatible with join validation), `model/__pycache__/*.pyc`, `benchmarks/results/profiles/*.cpuprofile` (embedded local paths; regenerable), `benchmarks/STAGE7_PROGRESS.md` (superseded) |
| 18 | Stage 5–7 reports address you personally | Moved to `docs/history/` |
| 19 | Comment-only brand fixes ("MESH" → "MINIMOL" in docstrings) | Wire identifiers (`"MESH"` magic bytes, `MESH-` room codes, `MeshPipeline`, channel label) deliberately left alone: renaming would change the protocol |
| 20 | **The room code was never visible to the host.** `stage()` in `Frontend/main.js` hid the whole `#panes` container while connecting, and the code box lives inside it, so after **Create room** the screen only said "Waiting for another device…". The host had nothing to tell the other device. This is present in the original frontend commit and in HEAD, so it is older than Stage 8. The Stage 6 browser test could not see it because it reads `#codeOut`'s text, which exists even when hidden | Fixed: while waiting as host the code and Copy button stay visible and the Create/Join controls are hidden |
| 21 | No test clicked only what a user can see | New `tests/browser/ui_user_flow.js` (13 checks). Mutation-checked: with the old behaviour restored it fails on exactly the two code-visibility checks |
| 22 | Output is random bytes (including unprintable characters) and nothing in the UI said why | Added an "untrained weights" note and a truthful placeholder on the Inference tab |
| 23 | Docs omitted that the workspace opens on **Overview** and the prompt is on the **Inference** tab | README, DEMO_SCRIPT and TESTING corrected |
| 9b | The Copy fallback from row 9 did not trigger when the clipboard API existed but was refused (headless: "Copy failed") | Now falls back to `execCommand` in both cases; checked from `localhost` and from a non-secure `http://<ip>` origin ("Copied") |

**Left alone on purpose:** `Frontend/dc-test.js` and `Frontend/stage5-test.js` are loaded by the real page; they are console diagnostics (`meshTest`, `meshStage5`), useful for physical-device evidence, so they stay and are documented. `/fixtures` is served publicly by the dev server (test data only). No benchmark result file was modified. No test was removed or weakened. No model, protocol, WebRTC or Socket.IO change.

## 3. Final project structure

```
minimol/
├── .gitattributes  .gitignore  LICENSE (MIT)  requirements.txt
├── README.md  ARCHITECTURE.md  INSTALLATION.md  TESTING.md  BENCHMARKS.md
├── DEMO_SCRIPT.md  TECHNICAL_INTERVIEW_PREPARATION.md  CONTRIBUTING.md  STAGE8_STATUS.md
├── BENCHMARK_RESULTS.json          (Stage 4, Windows; see BENCHMARKS.md §13)
├── Frontend/        main.html, main.js, main.css, connection_state.js, dc-test.js, stage5-test.js
├── engine/          inference_engine.js, mesh_pipeline.js
├── protocol/        mesh_protocol.js
├── model/           transformer.py, export_weights.py, export_stage5_reference.py, verify_reference.py
├── room-server/     server.js, package.json, package-lock.json, test/signalling.test.mjs
│                    (public/weights/ generated, git-ignored; node_modules/ git-ignored)
├── tests/           test_tensor/protocol/js_engine/distributed_inference/stage5_pipeline/
│                    stage6_reliability/stage7_benchmarks.js, benchmark.js, fixtures/, browser/
├── benchmarks/      run.js, bench_browser.js, profile_workload.js, analyze_profile.js, lib/, results/
└── docs/history/    STAGE5/6/7_STATUS.md, DEMO_GUIDE_stage4_outdated.md
```
The publishable tree is 76 files and about 1.8 MB (the largest file is the 797 KB PyTorch reference fixture).

## 4. Verification record

**Verified in Stage 8 from a clean copy of the publishable files, following INSTALLATION.md** (`npm install`, `export_weights.py`, `export_stage5_reference.py`, then tests):

| Check | Result |
|---|---|
| Weight export (sizes 22,473,216 / 21,689,856 / 44,163,072 B) | OK |
| Regenerated PyTorch fixture vs committed fixture | **Byte-identical** (deterministic) |
| `verify_reference.py` | Pass |
| `test_tensor` / `test_protocol` / `test_js_engine` / `test_distributed_inference` | Pass (38 checks in the tensor suite) |
| `test_stage5_pipeline` / `test_stage6_reliability` / `test_stage7_benchmarks` | 50 / 24 / 46 passed, 0 failed |
| `cd room-server && npm test` (new) | 15 passed, 0 failed |
| `node benchmarks/run.js --profile quick` | Numerical gate passed (max diff from PyTorch 9.7e-7 activations, 8.9e-7 logits), 0 failures |
| Server routes (page, scripts, engine, protocol, weights, Socket.IO client) | All HTTP 200 |
| 52 relative links and anchors across all markdown | 0 broken |
| 6 Mermaid diagrams in ARCHITECTURE.md | Parse with the real Mermaid parser (found and fixed one syntax error). **Parsed, not visually rendered**: check them on GitHub |
| Secret / key / email / absolute-path scan of publishable files | Nothing found (the only identity in files is the handle in `LICENSE`) |
| Missing-weights, port-in-use and missing-Chrome messages | Each confirmed to print the intended message |
| `tests/browser/stage6_two_page.js` | 21 passed, 0 failed (headless Chromium, one machine; re-run after the `main.js` changes) |
| `tests/browser/ui_user_flow.js` (new) | 13 passed, 0 failed |
| Real UI flow from `http://192.0.2.2:3000` (non-secure context, same machine) | Connected, both INFERENCE_READY in ~0.9 s, generation Done at ~7.8 tokens/s, Copy button works |

**Not verified in Stage 8:**

| Item | Why |
|---|---|
| In-page `meshStage5.validate()` (27 of 27) | Reported in Stage 5, not re-run |
| Browser benchmark (`bench_browser.js`) | Not re-run |
| Two physical devices, Firefox, Safari, mobile, different networks, TURN, plain-http on a phone | Never run by anyone |
| The CPU-only PyTorch install command in INSTALLATION.md | Index not reachable from here (the CUDA wheel route was tested) |
| Node versions other than 22, Windows/macOS | Not run (`engines` says `>=18`, which is an expectation) |
| Your own understanding (the project's real finish line) | Only you can show it: see §8 |

## 5. Feature status

**Implemented and tested at the stated level:** room matchmaking, Socket.IO signalling relay, WebRTC peer connection + ordered DataChannel (one-machine browser run), binary tensor protocol with checksum/validation/chunking, 2-way model partition with fingerprint handshake, distributed JS inference (bit-identical to local; ≤1.1e-6 from PyTorch), timeouts/duplicate handling/`dispose()`/bounded reconnect, benchmark harness.
**Dev aid only:** "Open workspace in dev demo mode" (canned text, labelled fake in the UI).
**Not implemented:** training, KV cache, TURN, auth, GPU/WASM, >2 devices, sampling beyond greedy.
**Untested:** everything in the "Not verified" table above.

## 6. Benchmark summary (all same-machine, shared 1-core VM)

- Local vs distributed latency: **no measurable difference** at T=8/35/128 (CI of the median difference contains 0); a sequential pipeline cannot beat local.
- Protocol overhead 0.10–0.90% of a request; compute 99%+; activation packet = `48 + 1536·T` bytes.
- Optimisation kept: blocked matmul, 2.0–2.4× (interleaved, bit-identical); generation 6.1 → 13.1 tokens/s local. Attention interchange reverted.
- Loopback DataChannel ~15–25 MB/s in headless Chromium: **not** a network figure.
- **No physical-device or real-network numbers exist.**

## 7. Known limitations

Untrained random weights (meaningless text) · same-machine testing only · no TURN (some networks cannot connect) · main-thread compute freezes the UI · no KV cache · greedy decoding, byte vocabulary, 512 context · two devices, one request at a time · no auth/rate limiting · Adler-32 is not security · ERROR-attribution gap L1 (a stray ERROR from an old half-received message can fail the next request once; fix needs a protocol change) · room codes are convenience, not secrets.

## 8. Interview preparation summary

`TECHNICAL_INTERVIEW_PREPARATION.md` holds: a 30 s/2 min pitch; a why/talks-to/alternatives/trade-off/limit write-up for each of 7 components; about 45 Q&As on networking, distributed systems, ML, software engineering and performance, each tied to files and measured numbers; 7 hard follow-ups with honest answers; 10 self-test exercises; a terms cheat-sheet.
**Your remaining job:** the Stage 8 goal says the project is finished when implementation, documentation, tests *and your understanding* align. I can write the material but cannot check that you understand it. Do exercises 1–10 yourself, then try explaining the architecture to someone without the docs. Two things worth being able to tell from memory: the benchmark-ordering story (BENCHMARKS.md §2–§3) and why a pipeline split cannot speed up a single request.

## 9. What you need to do next (nothing below has been done for you)

1. **Review the diff**, then commit in logical pieces so history is readable. Suggested order:
   1. `git add --renormalize .` then commit the repository hygiene (`.gitignore`, `.gitattributes`, untracking `node_modules`, deletions).
   2. Commit Stage 5–7 code: `engine/mesh_pipeline.js`, `protocol/`, `Frontend/`, `model/`, `tests/`, `benchmarks/`, `room-server/` (that commit is the one that makes the repository complete).
   3. Commit the documentation.
2. **Git history contains a personal email** (`pratyushk101608@gmail.com`) in older commit authors. You chose to leave history alone; for new commits you may prefer GitHub's noreply address: `git config user.email "<id>+<username>@users.noreply.github.com"`.
3. **LICENSE** is MIT, copyright "2026 PratyushK0_0" (confirm that is the name you want shown). MIT permits anyone to use, modify and sell the code provided the notice is kept, with no warranty.
4. **Replace `<your-repository-url>`** in README.md and INSTALLATION.md.
5. **Physical-device test** (TESTING.md §7) and, if it works, record the real result in TESTING.md/BENCHMARKS.md under a "physical device" label. Until then say "two browser instances on one machine".
6. **Run the two browser tests once on your own machine** (TESTING.md §5) and click through the app yourself in a normal (non-headless) browser: headless Chromium on Linux is not the same as your setup.
7. **Screenshots:** none exist. Capture the connected state and an output with metrics during your dry run and add them to the README.
8. Look at the Mermaid diagrams on GitHub after pushing.
9. Optional later: a CI workflow running the Node tests (needs weights generated in CI), and a root `package.json` with scripts.

## 10. GitHub description and topics

**Description (under 350 characters):**
> Browser-based distributed transformer inference: an 11M-parameter byte-level GPT split across two devices, with WebRTC DataChannel tensor transfer, Socket.IO signalling, a custom binary protocol and a benchmark harness. Untrained demo weights; tested same-machine.

**Topics:** `webrtc` `datachannel` `socket-io` `distributed-inference` `transformer` `peer-to-peer` `model-partitioning` `browser-ml` `edge-ai` `binary-protocol` `javascript` `nodejs` `pytorch` `benchmarking`

## 11. Demonstration recording checklist (full version in DEMO_SCRIPT.md §5)

Choose variant A (two windows, one computer) or B (physical devices, only if tested) · fresh clone + weights + server running · hard-reload both pages · fixed prompt `MESH`, one rehearsal · never click "dev demo mode" · show server, create, join, partition download, live inference with metrics, optionally kill one side · show 3 engineering points + one benchmark chart/table · state the untrained model and same-machine limits out loud · take screenshots during the dry run.

## 12. Readiness assessment

| Dimension | Assessment |
|---|---|
| **Reproducibility** | **Strong.** Verified from a clean copy of the publishable files; deterministic weights proven by a byte-identical fixture |
| **Documentation accuracy** | **Good, with a lesson.** Claims are labelled by evidence level; several of my own draft claims were caught and corrected against the code (session IDs are time-based not random, the ICE config location, a wrong troubleshooting row) |
| **Test coverage** | **Good** for logic, protocol, pipeline, signalling and the full UI journey in headless Chromium; **gaps:** no CI, no non-Chromium browsers, only one machine |
| **Benchmark honesty** | **Strong.** Statistical method, noise floor, a reverted optimisation and a non-reproducing result are all documented |
| **Physical-device evidence** | **None.** This is the largest gap between "works" and "demonstrated" |
| **Publication readiness** | **Ready once you commit** (§9.1–9.4). The current HEAD alone is not publishable |
| **Interview readiness** | Material ready; **depends on you completing the self-test** |

**Bottom line:** MINIMOL is a technically defensible portfolio project as "a two-browser distributed inference pipeline, verified on one machine with an untrained model". It is not yet defensible as "works across devices on real networks", and the documentation never says it is.
