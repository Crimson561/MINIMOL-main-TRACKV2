# STAGE 6 — Reliability, Error Handling, Recovery (status)

Last updated 2026-11-15. Everything under "Results" was actually executed; anything not run says so.

## 1. Reliability audit (Stage 5 as found)

Found by reading the source; the three marked **(reproduced)** were also confirmed with throwaway probes against the Stage 5 code.

| # | Gap in Stage 5 | Where | Stage 6 fix |
|---|---|---|---|
| 1 | **(reproduced)** Channel closes while A waits for B's HELLO: `infer()` settled only after the full timeout (probe: closed at 100 ms, rejected at 3005 ms). | `dropPipeline` → `infer` | `PipelineBase.dispose()` rejects waiters at once |
| 2 | **(reproduced)** Session-`"0"` ERROR matches *any* pending request; B emitted one ERROR per leftover chunk after a bad chunk, so stragglers could fail the next healthy request. | `_onPeerError`, `ChunkReassembler` | One ERROR per broken message (see limitation L1) |
| 3 | **(reproduced)** A half-received chunked message was held (up to 8 MiB) until some other message arrived. | `ChunkReassembler` | Inactivity expiry `reassemblyTimeoutMs` |
| 4 | After `dc.onclose`: `S.live` stayed `true`, stage stayed "connected", `S.dc`/`S.pc` never cleared. | `wire()` | State machine + `closePeer()` |
| 5 | pc `disconnected` only logged; `closed` ignored. | `makePC` | Grace timer then `linkLost()` |
| 6 | Socket.IO `disconnect` only logged; server never told the other device when its peer left. | `connect`, `server.js` | `peer-left` event; explicit rules |
| 7 | 30 s "establishing" and 6 s room-code timers never cleared: an old session's timer could fail a new one. | `makePC`, `createBtn` | `later()/cancel()/clearTimers()` |
| 8 | `dcTransport` always used the *current* `S.dc`: an old pipeline could write into the new session's channel. | `dcTransport` | `makeTransport(gen)` |
| 9 | No duplicate-init guard: second `peer-joined` re-ran `makePC(true)` and overwrote `S.pc` unclosed; B reused a stale pc for a new offer. | handlers | State guard + `makePC` closes the old pc first |
| 10 | ICE candidate arriving before `setRemoteDescription` was dropped; `.catch(()=>{})` hid it. | `E.ice` handler | Buffer + `flushIce()`; errors logged |
| 11 | `fail()` in the workspace screen was invisible except via a 2 s render poll. | `stage()` | `fail()` returns to the matchmaker failure box |
| 12 | `connect()` failure returned normally, so `createBtn` went on to use a null socket. | `connect` | Now throws |
| 13 | Run button fell back to canned *demo* text if a live session lost its link (after Stage 6 made `S.live` honest). | `runBtn` | Demo only when `S.demo` |

Already present and **kept unchanged**: `MeshError` codes, the Stage 5 ERROR packet, packet/tensor/checksum validation, fingerprint check, session+seq matching, duplicate/late reply ignoring, backpressure, message-size limit, 8 MiB reassembly cap. No second error protocol was created.

## 2. Files

**Created**
- `Frontend/connection_state.js` — pure state machine + `reconnectPlan()`
- `tests/test_stage6_reliability.js` — 24 tests (unit + Node simulation)
- `tests/browser/stage6_two_page.js` — optional scripted two-page browser run (needs Playwright; not a project dependency)
- `STAGE6_STATUS.md`

**Modified** (targeted changes only)
- `engine/mesh_pipeline.js` — `dispose()`, reassembly expiry, `_assertAlive()`
- `protocol/mesh_protocol.js` — `ChunkReassembler`: `poisonId`, `suppressed`, `reset()` clears both (additive)
- `room-server/server.js` — emits `peer-left`, closes the room when a member leaves
- `Frontend/main.js` — state machine, generations, timers, `closePeer/teardown/fail/reset`, `linkLost/startReconnect`, ICE buffering, `makeTransport(gen)`
- `Frontend/main.html` — one `<script src="connection_state.js">` line
- `TESTING.md`, `ARCHITECTURE.md` — new Stage 6 sections

Not touched: frontend design/theme, inference engine, model, binary protocol format, dependencies, `stage5-test.js`, `dc-test.js`.

## 3. What changed and why

1. **State machine** (`connection_state.js`). One state replaces five scattered flags. Events with no arrow are refused, so a late callback cannot move the app to a nonsense state.
2. **Generations** (`S.gen`). `closePeer()` bumps it; every callback captures its generation and returns if stale. Old sessions cannot touch new ones.
3. **Timers** all go through `later()`, so cleanup cancels every one of them.
4. **Idempotent cleanup**: `closePeer()` (pc, dc, pipeline, timers, ICE buffer) and `teardown()` (+ socket). Handlers are detached *before* `close()` so our own close does not trigger our own failure path.
5. **Link loss**: dc `close`, pc `failed`, pc `disconnected` longer than 8 s (0 s if signalling already said the peer left) → `LINK_LOST` → `RECONNECTING`.
6. **Bounded recovery**: A re-offers up to 2 times (1 s, 2 s backoff, 15 s per attempt); B waits 20 s per attempt; then `FAILED` with a reason. The budget refills only after a *full successful inference*, because "reconnected" is not "working".
7. **Signalling**: `reconnection:false` (a silent Socket.IO reconnect yields a socket that is in no room). Losing signalling before the DataChannel exists fails the session; after it exists inference continues but reconnection becomes impossible.
8. **`dispose()`** on pipelines: rejects the request in flight and all waiters *now*, clears timers, ignores later messages, silences its event callback.
9. **Reassembly expiry** + **one ERROR per broken message**.
10. `runLive()` refuses unless `INFERENCE_READY`, with a message that says why.

## 4. Results (run 2026-11-15)

| Suite | Kind | Result |
|---|---|---|
| `test_tensor.js` | Unit | 38 passed |
| `test_protocol.js` | Unit/simulation | PASSED |
| `test_js_engine.js` | Node | PASSED |
| `test_distributed_inference.js` | Node simulation | PASSED |
| `test_stage5_pipeline.js` | Node simulation | **50 passed, 0 failed** (unchanged) |
| `model/verify_reference.py` | Python | PASSED |
| `test_stage6_reliability.js` | Unit + Node simulation | **24 passed, 0 failed** |
| `tests/browser/stage6_two_page.js` | Scripted, two headless Chromium pages, one machine, real Socket.IO + WebRTC | **21 passed, 0 failed** |
| Physical devices | — | **NOT RUN** |

Stage 6 test → required scenario: (1) channel closes during inference (3 tests) · (2) timeout · (3) invalid activation (NaN / bad shape / bad checksum) · (4) duplicate response · (5) late response · (6) incomplete chunk expiry (+ no premature expiry, + no leaked timer, + broken chunk = one error) · (7) peer vanishes mid-request · (8) fingerprint mismatch · (9) reconnection → valid new session · (10) cleanup called repeatedly.

**Mutation check:** three deliberate bugs injected into a scratch copy (timer not cleared on settle; `dispose()` not rejecting waiters; reassembly timer never armed) were each caught by ≥1 test.

**Browser run highlights** (one machine, so timings are optimistic): channel closed mid-inference → request failed in 36 ms, both devices reconnected over a new peer connection in ~1 s, next inference worked; B's tab closed → A ended in `FAILED` after ~5 s with "the other device has left the room"; with B refusing offers, A stopped after exactly 2 attempts in ~9 s.

## 5. Remaining limitations (be honest about these)

- **L1.** The session-`"0"` ERROR hole is *reduced, not closed*. A stray ERROR from an old broken transfer that arrives after that request timed out can still fail the *next* request (B cannot know the session of a chunk it could not reassemble, and the Stage 5 ERROR packet cannot carry a message id without a protocol change, which was out of scope). Now at most one stray per broken message. The failure is reported (not silent) and a retry works.
- **L2.** Reconnection builds a **new** peer connection; there is no ICE restart. Fine for this app, slower than a restart.
- **L3.** No TURN server: devices behind strict NATs may never connect, and no code can fix that.
- **L4.** When signalling says the peer left but the DataChannel still looks open, we wait for the pc/dc to actually fail (about 5 s on one machine; real networks may be slower, up to roughly 30 s for a silent power-off).
- **L5.** A's timeout does not tell B to stop computing; B's late reply is ignored by A.
- **L6.** If Socket.IO is lost *and* the link then drops, recovery is impossible by design and the session fails.
- **L7.** A reconnect keeps the loaded weights but not the generated text of an interrupted request: the partial output stays on screen and the user re-runs.
- **L8.** The state machine and `main.js` handlers are verified only by the scripted browser run in Chromium on one machine. No Firefox/Safari, no two real devices, no hand-driven two-tab run.
- **L9.** Weights are not in the repo; the tests need `python model/export_weights.py` and `python model/export_stage5_reference.py` first.

**Outside Stage 6 (reported, not fixed):** root `main.html` differs from `Frontend/main.html`; `server.js` still has leftover debug code (`ping-server`, a debug log line); `room-server/test-*.html` are old clients.

## 6. Questions you should be able to answer after Stage 6

1. Name the three layers that can each be "connected" or dead independently, and say which one inference depends on.
2. Why can a connected Socket.IO client coexist with a dead DataChannel?
3. What is the difference between pc `disconnected` and `failed`, and why do we wait before acting on the first?
4. What does ICE do, and what would a TURN server add?
5. Why does a dead peer connection need a *new* DataChannel/pipeline rather than reusing the old one?
6. What is the generation counter (`S.gen`) protecting against? Give a concrete sequence of events where it matters.
7. Why must `later()` exist instead of calling `setTimeout` directly?
8. Why is `dispose()` idempotent, and what two things would break if it were not?
9. Why must pending promises be rejected on channel close rather than left to time out?
10. Why is the infer timer cleared in `_settle`? What happens to memory and to the next request if it is not?
11. What makes a late or duplicate reply harmless, and which two fields are compared?
12. Why does a *failed* reassembler remember the id of the broken message?
13. Why is the reassembly timer an *inactivity* timer rather than a total-time limit?
14. Why does the retry budget refill only after a successful inference, and why are retries bounded?
15. Why are handlers detached before `pc.close()` / `dc.close()`?
16. Why does the state machine refuse events instead of throwing?
17. Which of the Stage 6 tests would still pass if the browser reported a failure we did not imagine? What does that say about simulations?

## 7. Manual testing (you do these; they are the evidence I could not produce)

Setup: export weights, `cd room-server && npm install && node server.js`, open `http://localhost:3000` in two tabs (or two devices, `http://<computer-ip>:3000`).
Useful console lines (type in the tab's DevTools console): `conn.state`, `S.gen`, `S.timers.size`, `S.reconnects`.

1. **Baseline**: create room in tab A, join in tab B, wait for "Live", run a prompt. Expect output. `conn.state` is `INFERENCE_READY` in both.
2. **Channel closed mid-run**: start a long prompt on A, then run `S.dc.close()` in A's console. Expect: run stops within ~1 s with "DataChannel closed", status "Reconnecting…", then "Live" again in both tabs; run again → works; `S.reconnects` back to `0`.
3. **Peer tab closed**: close tab B. Expect A to leave the workspace to the "Connection failed … other device left the room" box in a few seconds; Retry works; `S.timers.size` is `0`.
4. **Bounded retries**: in B's console `S.socket.off(CONFIG.events.offer)`, in A's `CONFIG.reconnectEstablishMs = 3000; S.dc.close()`. Expect exactly 2 reconnect attempts, then "Gave up after 2 reconnection attempt(s)".
5. **Signalling lost, link alive**: stop `server.js` mid-session. Expect inference to keep working. Then `S.dc.close()` in A: expect it to fail with "signalling connection is gone" (no endless retry).
6. **Cancel/duplicate**: click Create, then Cancel quickly, repeat several times; `S.timers.size` must be `0` and `conn.state` `DISCONNECTED`. In the console `connect("A")` twice: second must throw "already in progress".
7. **Physical devices** (not yet run by anyone): repeat 1–3 on two real devices; also switch one device's Wi-Fi off for ~15 s during a run and back on, and record what actually happens and how long it takes. Put the results here.

## 8. Learning summary

- **Connection layers and ICE**: signalling is a rendezvous, not the data path; ICE finds a NAT-traversing path; the DataChannel rides on it. `connectionState` `disconnected` may heal, `failed`/`closed` will not.
- **Failure is partial and independent**: each layer and each device can fail separately, so each needs its own detection and a single place that decides what it means (`linkLost`).
- **State machines**: one authoritative state, explicit transitions, refusal of impossible events.
- **Promises and request/response matching**: every pending request must eventually settle; session + seq identify which request a reply belongs to; late and duplicate replies are ignored, not applied.
- **Timeouts**: always bounded, always cancellable, always cleared when the thing they guard finishes.
- **Chunking and backpressure**: large tensors are split to fit the negotiated limit; incomplete transfers hold memory, so they expire; a broken message should cost one error, not one per frame.
- **Cleanup**: idempotent, ordered (state first, then resources), and detach handlers before closing. Generations stop old sessions interfering with new ones.
- **Testing honesty**: unit tests, simulations, scripted browser runs and physical tests prove different things; none substitutes for the next.

Revise later: ICE restart vs new peer connection; how browsers decide a DataChannel is closed after a silent network loss; microtask ordering when `dispose()` rejects several promises (why the pending request is settled before the waiter's error is observed).
