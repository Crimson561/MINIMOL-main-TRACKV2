# Stage 5 status (updated 2026-11-14)

## Implemented
- `engine/mesh_pipeline.js`: PipelineA / PipelineB (HELLO + weight fingerprint, session/seq matching, once-only processing on B,
  chunked send, ERROR propagation, timeouts). **Bug fixed:** removed `activation[7] += 1e-6`, which altered the real activation
  and made 4 of the 50 Node tests fail. Nothing else in this file was changed.
- `protocol/mesh_protocol.js` (additive): MeshError, encodeError, validateActivation, chunking, optional logits, FLAG_WANT_LOGITS.
- `model/export_weights.py`: Adler-32 per `.bin` in the manifest. `model/export_stage5_reference.py`: PyTorch fixture.
- `Frontend/main.js`: loads `mesh_pipeline.js`; verifies partition Adler-32 + size before use; DataChannel transport with
  readiness / `bufferedAmount` backpressure / `bufferedamountlow` / negotiated `sctp.maxMessageSize` / closure handling;
  `maybeInitPipeline()` (partition loaded AND channel open, runs once, either order); `onPacket` routes to the pipeline;
  `runLive()` uses `PipelineA.infer()`. The old manual activation/token exchange was removed.
- `Frontend/stage5-test.js` + script tag in `Frontend/main.html`: `meshStage5.validate() / local() / twoDevice()`.
- `room-server/server.js`: serves `/fixtures` and `/engine` (already present before this task; used unchanged).

## Test results actually obtained (2026-11-14)
| What | Result |
|---|---|
| Node: test_tensor / test_protocol / test_js_engine / test_distributed_inference | all pass |
| Node: test_stage5_pipeline | 50 passed, 0 failed (was 46/50 before the 1e-6 line was removed) |
| Python: verify_reference.py | passes (PyTorch full == PyTorch split, max diff 0.0; PyTorch vs NumPy 1.6e-6) |
| Headless Chromium, one page: `meshStage5.local()` | 17 passed, 0 failed |
| Headless Chromium, **two pages on one machine** (scripted): `meshStage5.validate()` on A | 27 passed, 0 failed, 0 skipped |
| Same two pages: B's `meshStage5.twoDevice()` | 4 passed, 2 skipped (A-only checks), 0 failed |
| Same two pages: UI run, prompt "MESH", 24 tokens | output equals Node full-model greedy output |
| Same: init-order race, tampered weights, channel closed mid-request | behaved as designed (see TESTING.md section 4) |

## NOT done / NOT verified
- **No manual test in your own browser** and **no physical-device test** has been performed. The two-tab result above came from a
  script driving two headless Chromium pages, not from you clicking in a real browser.
- Safari, Firefox, Android/iOS browsers, real networks (NAT/STUN, no TURN) are untested.
- Equality with PyTorch is **not bit-for-bit**: logits differ by up to ~1.1e-6 (tolerance 1e-5). Distributed vs local JS is bit-identical.
- No automated signalling-server tests exist (the earlier "40/40" claim was unsupported and removed).
- Main-thread compute (UI freezes during inference); no KV cache; see ARCHITECTURE.md "Known limits".

## Run
```
python model/export_weights.py && python model/export_stage5_reference.py
node tests/test_stage5_pipeline.js
cd room-server && npm install && npm start      # then follow TESTING.md "Manual test steps"
```
