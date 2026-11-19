# MINIMOL — Demonstration Script (3–5 minutes)

**Rule for this script: only say what has been tested.** Section 7 lists what you may and may not claim. Nothing in this file has been recorded yet, so every "show" step below is a plan, not a record of something already filmed.

## 0. Choose your demo variant before recording

| Variant | Setup | What you can honestly call it |
|---|---|---|
| **A. Two windows, one computer** (tested level) | `http://localhost:3000` in two browser windows side by side | "Two browser instances on one machine, talking over a real WebRTC DataChannel" |
| **B. Two physical devices** (not yet tested) | Computer + phone (or two computers) on one Wi-Fi, `http://<LAN-IP>:3000` | "Two devices": **only after** you have completed the checklist in [TESTING.md](TESTING.md#7-physical-device-checklist-not-yet-run) and it worked |

If B fails (firewall, Wi-Fi client isolation, NAT), film A and say plainly that it is one machine. Do not edit around the difference.

## 1. Timeline

| Time | Part | On screen | Say (talking points, in your own words) |
|---|---|---|---|
| 0:00–0:30 | **1. Introduction** | README title and the "Honest status" paragraph | MINIMOL splits one small transformer across two browsers. Big models do not always fit on one device, so a pipeline split puts early layers on one and later layers on another and passes the activations across. This is the systems side at toy scale. The model is untrained, so the text will be nonsense; the point is the engineering. |
| 0:30–1:15 | **2. Architecture** | The diagram in ARCHITECTURE.md §1, then the partition diagram §4 | Three pieces: a tiny Node server that only introduces the two browsers (Socket.IO signalling), a direct WebRTC DataChannel between them, and the model split after layer 2: Device A runs embeddings and layers 0–2, Device B runs layers 3–5 and the output head. After the channel opens the server carries no inference traffic. Activations are sent as raw float32 in a 48-byte-header binary packet. |
| 1:15–3:00 | **3. Live demonstration** | Two windows (or two devices) side by side, terminal with the server visible | See the step list below. |
| 3:00–4:15 | **4. Engineering** | (a) `engine/mesh_pipeline.js` / `protocol/mesh_protocol.js`; (b) BENCHMARKS.md §3–§5; (c) the test output | See section 3 below. |
| 4:15–4:45 | **5. Conclusion** | README "Limitations" | What was built, what you learned, what is missing: trained weights, KV cache, TURN, physical-device results; future work. |

## 2. Live demonstration steps (1:15–3:00)

1. Terminal: `cd room-server && npm start`. Point at the "listening" line.
2. Window 1 → **Find my other half** → **Create room**. Read the `MESH-XXXXX` code aloud. (Mention: MESH was the protocol's working name.)
3. Window 2 → **Find my other half** → enter the code → **Join room**.
4. Wait while each side downloads *its own* half of the weights. Say: each device fetches only its partition (22.5 MB and 21.7 MB), verified against checksums.
5. Show that both sides report a live session, then **Enter workspace →**.
6. In Device A: prompt `MESH`, **Run inference**. Show the output and the timing panel (A compute, B compute, round trip).
7. Say what the output is: random-weight text, deterministic, not meaningful.
8. (Optional, 20 s, strong evidence) Close Window 2 mid-run or reload it. Show Device A leaving the live state and reporting a reason instead of hanging.

**Do not click "Open workspace in dev demo mode".** It prints canned text and never contacts a peer.

Optional on-screen proof of correctness, if the console is visible: `await meshStage5.validate()` on the host (two-device checks are reported as SKIPPED unless a live pipeline exists, so read the result before you describe it).

## 3. Engineering segment (3:00–4:15)

Pick three, 20–25 seconds each:

1. **Tensor transfer.** Show the header layout (ARCHITECTURE.md §5). Raw float32 bytes, shape in the header, Adler-32 checksum, validation before compute, chunking only above the channel limit. Why binary: no JSON/base64 inflation.
2. **Proving correctness.** Split model equals full model: bit-identical in JavaScript, within about 1e-6 of PyTorch (tolerance 1e-5). Show `node tests/test_distributed_inference.js` passing.
3. **Benchmarks, honestly.** Distributed is *not* faster: A and B run one after the other, so it cannot beat local. Protocol overhead is under 1% of a request; compute is 99%+. The split buys memory per device, not speed. All numbers are from one shared 1-core VM and loopback.
4. **One interesting challenge (recommended): a measurement artefact.** An early benchmark showed distributed inference 44% *faster* than local at T=8, which is impossible for a sequential pipeline. The cause was run order (JIT warm-up favoured whichever mode ran second). The fix was interleaving local/distributed runs (ABBA) with a bootstrap confidence interval, and one "significant" result later failed to reproduce. Lesson: distrust a result you cannot explain. (Source: BENCHMARKS.md §2–§3.)
   *Alternative reliability story:* closing the DataChannel mid-request now rejects the in-flight request in 36 ms (browser run) instead of waiting for the 20 s timeout, because `dispose()` settles pending work immediately.

## 4. Conclusion points (4:15–4:45)

- Built: rooms and signalling, WebRTC DataChannel, binary tensor protocol, partitioned inference in plain JS, failure handling, a measured benchmark harness.
- Learned (use your own): what has to be true for two devices to agree (fingerprints, checksums, sessions), how to benchmark without fooling yourself.
- Limits: untrained model, no KV cache, main-thread compute, no TURN, no physical-device results yet.
- Next: train or load real weights, add a KV cache, WebGPU/WASM, a TURN server, measure on real devices.

## 5. Pre-recording checklist

- [ ] Fresh clone follows INSTALLATION.md; weights generated; `npm start` works.
- [ ] `cd room-server && npm test` and the Node tests pass (see TESTING.md).
- [ ] Two windows sized side by side; browser zoom 125%+; notifications off; dev tools closed unless needed.
- [ ] Both pages hard-reloaded (Ctrl+Shift+R) so no stale weights are cached.
- [ ] Variant B only: both devices open `http://<LAN-IP>:3000`, firewall allows the port, you have already done one full dry run on these devices.
- [ ] Pick the prompt in advance (`MESH`) and do one rehearsal to know roughly how long a run takes on your hardware.
- [ ] Have these tabs ready: README, ARCHITECTURE diagrams, `engine/mesh_pipeline.js`, BENCHMARKS.md.
- [ ] Screen recorder set to capture system audio off, microphone on, 1080p; do a 10-second test clip.
- [ ] Take screenshots during the dry run (connected state, output with metrics) and add them to the README under `docs/images/` if you want them: **no screenshots exist in the repository yet**.

## 6. If something goes wrong on camera

| Problem | Do |
|---|---|
| Devices never connect (variant B) | Switch to variant A and say so. Mention STUN-only/no TURN as the likely reason |
| "Weights REJECTED" | Hard-reload; if it persists, regenerate weights and restart the server |
| Output looks like garbage | Expected. Say it is untrained |
| Run is slow | Expected for a long prompt (no KV cache, main thread). Use a short prompt |
| Link drops | Good moment to show the reconnect/`FAILED` handling; do not hide it |

## 7. Claims: allowed vs not allowed

| You may say | Do not say (not tested or not true) |
|---|---|
| "Distributed across two browser instances using a WebRTC DataChannel" | "Runs on phones" / "across two physical devices" (until variant B works and is recorded) |
| "Split model output is bit-identical to the unsplit JavaScript model and within ~1e-6 of PyTorch" | "Matches PyTorch exactly" |
| "Protocol overhead is under 1% of a request on loopback" | "Network overhead is small" (no real network was measured) |
| "Same-machine benchmarks show no measurable latency win from splitting" | "Distributed inference is faster" |
| "Handles a peer disappearing: the request fails fast and the app reconnects or reports why" (browser, one machine) | "Production-grade reliability", "works on any network" |
| "The model is untrained, so the text is meaningless" | "It generates text" / "an AI model answers questions" |
