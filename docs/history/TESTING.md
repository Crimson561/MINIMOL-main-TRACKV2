# MINIMOL — Testing

This page says what is tested, how to run it, what you should see, and **what each kind of test does and does not prove**.

## 1. Evidence levels

| Level | What it proves | What it cannot prove |
|---|---|---|
| **Unit / Node simulation** | Logic is correct: packets, maths, state machine, pipeline protocol over a simulated link | Anything about real browsers or networks |
| **Signalling test** | The Socket.IO server enforces rooms and relays signals correctly | WebRTC itself |
| **Scripted browser run** (two headless Chromium pages, one machine, real Socket.IO + real WebRTC) | The frontend code works end to end in a real browser | Behaviour across a real network, other browsers, phones |
| **Physical devices** | The thing the project is about | — |

Current status (`Verified` = re-run by the Stage 8 audit on 2026-11-17, Linux, Node 22.23, Python 3.12.3, torch 2.15.0, numpy 2.5.4):

| Test | Level | Result | Status |
|---|---|---|---|
| `python model/verify_reference.py` | Python reference | PyTorch == NumPy engine decisions; split == full | **Verified** |
| `node tests/test_tensor.js` | Unit | 38 passed, 0 failed | **Verified** |
| `node tests/test_protocol.js` | Unit | passes ("protocol and data transfer integrity verified") | **Verified** |
| `node tests/test_js_engine.js` | Unit vs PyTorch | passes | **Verified** |
| `node tests/test_distributed_inference.js` | Simulation | passes (distributed == local) | **Verified** |
| `node tests/test_stage5_pipeline.js` | Simulation | 50 passed, 0 failed (~40 s) | **Verified** |
| `node tests/test_stage6_reliability.js` | Simulation | 24 passed, 0 failed (~15 s) | **Verified** |
| `node tests/test_stage7_benchmarks.js` | Unit (benchmark harness) | 46 passed, 0 failed (~20 s) | **Verified** |
| `cd room-server && npm test` | Signalling (real Socket.IO) | 15 passed, 0 failed | **Verified** (new in Stage 8) |
| `node tests/browser/stage6_two_page.js` | Scripted browser (two headless Chromium pages, one machine) | 21 passed, 0 failed | **Verified** (Stage 8; also passed in Stage 6) |
| `node tests/browser/ui_user_flow.js` | Browser, drives only **visible** controls like a person | 13 passed, 0 failed | **Verified** (new in Stage 8; mutation-checked: fails if the room code is hidden) |
| Same user flow from `http://192.0.2.2:3000` (not `localhost`; `isSecureContext` = false) | Browser, one machine, non-secure origin | connect, generate "Done", Copy button OK | **Verified once, manually** (Stage 8). Approximates a phone loading the page over plain http; it is **not** a phone |
| In-page `meshStage5.validate()` in two tabs | Two tabs, one computer | 27 of 27 (17 local + 10 two-device) | Reported in Stage 5, not re-run in Stage 8 |
| Two physical devices | Physical | — | **NOT RUN** |
| Firefox, Safari, mobile browsers | — | — | **NOT RUN** |
| TURN / restrictive networks | — | — | **NOT RUN** |

## 2. Prerequisites

Complete [INSTALLATION.md](INSTALLATION.md) first, in particular `python model/export_weights.py` (the tests need `room-server/public/weights/`) and `cd room-server && npm install` (the signalling test needs `socket.io-client`). Without weights the Node tests stop with `Weights missing. Run: python model/export_weights.py`.

## 3. Run everything

From the repository root (Linux/macOS shell; on Windows run the lines individually):

```bash
python model/verify_reference.py
for t in test_tensor test_protocol test_js_engine test_distributed_inference \
         test_stage5_pipeline test_stage6_reliability test_stage7_benchmarks; do
  node tests/$t.js || break
done
(cd room-server && npm test)
```

Each Node test exits 0 on success and non-zero on failure. Total run time is about 2 minutes on a single-core VM.

## 4. What each test covers

| Test | What it checks |
|---|---|
| `verify_reference.py` | The pure-NumPy engine reproduces PyTorch's next-byte decisions; the A+B split equals the full model |
| `test_tensor.js` | Tensor serialization round-trips bit-identically (all shapes/edge cases), both directions; malformed tensors are rejected |
| `test_protocol.js` | Packet header layout, checksum, corruption detection, activation transfer integrity |
| `test_js_engine.js` | JS engine vs the committed PyTorch fixture (`tests/fixtures/stage5_reference.json`); tolerance 1e-5 |
| `test_distributed_inference.js` | Device A → packet → Device B gives the same answer as one local model |
| `test_stage5_pipeline.js` | HELLO handshake, session/sequence matching, once-only processing, chunking and reassembly, validation, error packets, timeouts, wrong-weights rejection |
| `test_stage6_reliability.js` | Connection state machine, bounded retry plan, `dispose()` rejecting in-flight work, chunk expiry |
| `test_stage7_benchmarks.js` | Statistics code and measurement wrappers (percentiles, bootstrap CI, restoring wrapped functions) |
| `room-server/test/signalling.test.mjs` | Room create/join, 2-device limit, relay of offer/answer/ICE, dropped malformed or out-of-room signals, `peer-left`, room closure |
| `tests/browser/stage6_two_page.js` (optional) | Two Chromium pages through the real server and a real DataChannel: connect, infer, kill channel mid-request, reconnect, tab close |
| `tests/browser/ui_user_flow.js` (optional) | The whole user journey through visible controls: room code shown, join, both ready, enter workspace, Inference tab, generation Done, no page errors |

**Numerical correctness.** The fixture was produced by PyTorch (`model/export_stage5_reference.py`). Regenerating it on a different machine in Stage 8 gave a **byte-identical** file, so the weights and reference are deterministic. JS vs PyTorch maximum differences were 9.7e-7 (activations) and 8.9e-7 (logits) in the Stage 8 benchmark gate, within the 1e-5 tolerance; JS distributed vs JS local is bit-identical.

## 5. Optional: browser tests

Need Playwright with Chromium, which is **not** a project dependency:

```bash
npm install --no-save playwright && npx playwright install chromium   # one-off, from the repository root
(cd room-server && npm install)
node tests/browser/ui_user_flow.js        # 13 checks; starts its own server on a spare port
node tests/browser/stage6_two_page.js     # 21 checks; starts its own server; port 3000 must be free
```

`ui_user_flow.js` exists because `stage6_two_page.js` reads the app's internal state, and an internal check cannot notice a control that is in the DOM but invisible: that is exactly how a hidden room code (a real bug found in Stage 8) went unnoticed. Prefer tests that click what a user can see. Both scripts launch Chromium with `--disable-features=WebRtcHideLocalIpsWithMdns` so two pages on one machine connect directly. If Node cannot find `playwright`, install it as above or set `NODE_PATH` to a folder containing it.

## 6. Manual two-tab test (no physical devices needed)

1. `cd room-server && npm start`
2. Open `http://localhost:3000` in two tabs (or one normal and one private window).
3. Tab 1: *Create room*. Tab 2: *Join* with the displayed `MESH-XXXXX` code.
4. Wait until both show the pipeline as ready (each loads its own half of the weights).
5. On both tabs click **Enter workspace →**. In tab 1 (Device A, the host) open the **Inference** tab (the workspace opens on *Overview*), type a prompt such as `MESH` and click **Run inference**. Expect a few bytes of **meaningless text** (possibly with unprintable characters): the weights are random, and the page says so.
6. Optional built-in diagnostics (browser developer console, once both devices are connected): `await meshTest.all()` (ping/pong and a 100 KB binary echo, from `Frontend/dc-test.js`) and `await meshStage5.validate()` (local checks plus, with a live pipeline, two-device checks; checks that cannot run are reported as SKIPPED, never as passed; from `Frontend/stage5-test.js`). These are the same helpers behind the Stage 5 "27 of 27" result and are the quickest way to collect evidence on physical devices.

## 7. Physical-device checklist (not yet run)

Do this before claiming the system works across devices. Record results in `STAGE8_STATUS.md` or a new section of `BENCHMARKS.md`, labelled as physical-device results.

1. Computer and phone on the same Wi-Fi. Start the server; find the computer's LAN IP (`ipconfig` / `ip addr`).
2. Allow inbound TCP 3000 in the computer's firewall.
3. Open `http://<LAN-IP>:3000` on both devices.
4. Create / join / infer as in section 6. Record: device models, OS and browser versions, network type, whether it connected, and the per-request numbers shown in the app (`computeMsA`, `computeMsB`, `rttMs`, total).
5. Switch one device's Wi-Fi off for ~15 s and back on; record what happens and how long it takes.
6. Repeat with the two devices on **different** networks (for example one on mobile data). Expect this to fail without a TURN server; note the result either way.

### HTTPS and secure contexts

`getUserMedia` (camera/microphone) requires HTTPS, but MINIMOL uses neither, and the WebRTC specification does not require a secure context for a DataChannel. A search of the frontend found exactly one secure-context-only API: `navigator.clipboard` in the **Copy code** button. Over plain `http://<LAN-IP>:3000` it does not exist, so the click would have thrown. Stage 8 added a fallback (`document.execCommand("copy")`, also used when the clipboard API is refused, and a "Copy failed" label if both fail). It was checked in Chromium from both `localhost` and a non-secure `http://<ip>` origin (label changed to "Copied"); not on a phone.

Whether a phone connects over plain `http://<LAN-IP>:3000` is **expected to work but unverified**. If a browser refuses something, serve over HTTPS (for example with a local certificate from `mkcert`) and open the `https://` address on both devices.

## 8. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `Weights missing. Run: python model/export_weights.py` | Weights are generated, not stored in git | Run the export; see INSTALLATION.md |
| `Cannot find module 'socket.io-client'` in `npm test` | Dev dependency not installed | `cd room-server && npm install` |
| Signalling test: `server exited early` / `EADDRINUSE` | Random port collision (rare) | Re-run |
| `Cannot find module 'playwright'` | Optional dependency | See section 5 |
| `stage6_two_page.js` fails to start | Port 3000 in use | Stop whatever uses it |
| Stage 5 test slow (~40 s) | It includes deliberate timeouts | Expected |
| Tests pass but Python checks fail on import | `torch` missing | `pip install -r requirements.txt` (large; see INSTALLATION.md) |
