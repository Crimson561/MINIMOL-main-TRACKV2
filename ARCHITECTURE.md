# MESH — System Architecture & Protocol Specification

## 1. Architectural Philosophy

MESH implements authentic pipeline model parallelism across two edge devices running mobile web browsers.
No central relay, server proxy, or cloud inference API is used during generation.

```
+---------------------------------------------------------------------------------------------------------+
|                                        FULL MODEL: 11,040,768 PARAMS                                    |
+----------------------------------------------------+----------------------------------------------------+
|               PHONE A (PARTITION A)                |               PHONE B (PARTITION B)                |
|               5,618,304 PARAMS (FP32)              |               5,422,464 PARAMS (FP32)              |
+----------------------------------------------------+----------------------------------------------------+
| 1. Byte Token Embedding: wte [256, 384]            | 1. Transformer Block 3 (Attention + MLP)           |
| 2. Learned Positional Embedding: wpe [512, 384]    | 2. Transformer Block 4 (Attention + MLP)           |
| 3. Transformer Block 0 (Attention + MLP)           | 3. Transformer Block 5 (Attention + MLP)           |
| 4. Transformer Block 1 (Attention + MLP)           | 4. Final LayerNorm: ln_f [384]                     |
| 5. Transformer Block 2 (Attention + MLP)           | 5. Output Head: lm_head [256, 384] (Untied)       |
|                                                    | 6. Deterministic Greedy Argmax Decoder             |
+----------------------------------------------------+----------------------------------------------------+
                          |                                                    ^
                          |  Activations: [1, T, 384] (Float32Array)           |
                          +----------------------------------------------------+
                                 Direct WebRTC DataChannel (No Server)
                          <----------------------------------------------------+
                             Selected Next Token: uint8 (0..255)
```

---

## 2. Transformer Specification & Layer Details

### Dimensions
- Vocabulary $V = 256$
- Hidden Dimension $D = 384$
- Number of Layers $L = 6$
- Attention Heads $H = 6$
- Head Dimension $d_k = D / H = 64$
- Feed-Forward Dimension $D_{ff} = 1536$ ($4 \times D$)
- Context Length $T_{max} = 512$
- LayerNorm Epsilon $\epsilon = 10^{-5}$

### Attention Mechanism
GPT-style causal multi-head self-attention:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
where causal mask $M_{i,j} = 0$ for $j \le i$, and $-\infty$ for $j > i$.

### Feed-Forward Network
$$\text{MLP}(x) = \text{Proj}(\text{GELU}(\text{FC}(x)))$$
with GELU tanh approximation:
$$\text{GELU}(x) = 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$

### Partition Cut Point
The model is partitioned cleanly between Block 2 and Block 3:
- **Partition A boundary output**: Tensor shape `[1, T, 384]` in FP32.
- **Partition B boundary input**: Tensor shape `[1, T, 384]` in FP32.
Because the boundary is an intermediate residual stream, the numerical outputs of Partition A followed by Partition B are mathematically identical to the full unpartitioned model.

---

## 3. WebRTC Binary Framing Protocol

Inference tensors and control messages travel over an ordered, reliable WebRTC DataChannel (`pc.createDataChannel("mesh", { ordered: true })`).

### Packet Header Layout (48 bytes)

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Magic "MESH" (0x4D455348)               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|    Version    |    MsgType    |             Flags             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                   Session ID (uint64, 8 bytes)                +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Sequence Number                         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   Data Type   |  Tensor Rank  |           Reserved            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Dimension 0                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Dimension 1                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Dimension 2                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Dimension 3                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Payload Byte Length                       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Adler-32 Checksum                         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Payload Bytes                          |
|                             ...                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### Message Types:
- `0x01` (`MSG_ACTIVATION`): Intermediate float32 activation tensor sent from Phone A to Phone B.
- `0x02` (`MSG_TOKEN`): Next token byte selected by Phone B sent back to Phone A.
- `0x03` (`MSG_CONTROL`): Generation control commands (`START_GENERATION`, `STOP`, `DONE`).
- `0x04` (`MSG_ERROR`): Fatal error signal.

### Security and Validation:
1. Every packet is validated for the `0x4D455348` magic prefix.
2. The payload length is strictly checked against the received buffer bounds.
3. Adler-32 checksums detect wire corruption.
4. Tensor dimensions are verified before memory allocation.
5. Out-of-order sequence numbers or unexpected session IDs are rejected.

---

## 4. Signalling & Peer Connection Lifecycle

1. **Room Creation**: Phone A sends `create-room` via Socket.IO. The server issues a room code (e.g. `MESH-7DK2P`).
2. **Room Joining**: Phone B sends `join-room` with the code. The server notifies Phone A (`peer-joined`).
3. **Offer/Answer Handshake**:
   - Phone A creates an `RTCPeerConnection` with STUN (`stun:stun.l.google.com:19302`).
   - Phone A calls `createDataChannel("mesh")` and `createOffer()`.
   - The SDP offer is relayed through Socket.IO to Phone B.
   - Phone B calls `setRemoteDescription`, creates an SDP answer, and sends it back to Phone A.
   - ICE candidates are exchanged.
4. **DataChannel Open**:
   - WebRTC connection state changes to `connected`.
   - DataChannel `onopen` fires on both devices.
5. **Decoupled Signalling**:
   - The signalling server is idle during inference.
   - All tensor activations and generated tokens travel solely through the direct DataChannel.

---

## 5. Stage 5 — Activation-transfer pipeline (as implemented)

Files: `engine/mesh_pipeline.js` (coordination), `protocol/mesh_protocol.js` (packets, chunking, validation), `engine/inference_engine.js` (the maths), `Frontend/main.js` (wires it to the real DataChannel), `Frontend/stage5-test.js` (console validation).

### One inference request
```
Device A (PipelineA.infer)                                Device B (PipelineB)
  tokens -> partition A (embeddings + layers 0-2)
  activation [1, T, 384] float32
  ACTIVATION packet {sessionId, seq=1}  -- chunked if larger than the channel limit -->
                                                          reassemble, decodePacket (Adler-32), validateActivation,
                                                          once-only check, partition B (layers 3-5 + ln_f + lm_head)
  <-- TOKEN packet {same sessionId, same seq} (+ logits only if A asked)   or   ERROR packet {code, message}
  accepts the reply only if sessionId and seq match the request in flight
```
`PipelineA.infer()` runs one forward pass (one next byte). Text generation is a loop in `main.js` `runLive()` that calls `infer()` again with the longer token list; there is no KV cache, so every step recomputes the whole sequence.

### Model identity (HELLO)
Each device sends a CONTROL packet `HELLO {role, loaded, fingerprint}` after its pipeline is created. The fingerprint is a string built from the manifest (architecture, sizes, seed, the cut after layer 2, and the Adler-32 of **both** partitions). `PipelineBase` does not reply to HELLO, so both sides send their own. A refuses to run until it has B's valid HELLO, and B refuses activations until it has A's.

### Weight verification
`export_weights.py` writes `checksum_adler32` (`zlib.adler32`) for each `.bin` into `weights_manifest.json`. The browser calls `MeshPipeline.verifyPartitionBuffer(buffer, manifest, "a" | "b")` (same Adler-32 as `MeshProtocol.adler32`, plus a byte-size check) **before** building an engine. On mismatch it throws `MODEL_MISMATCH`; `main.js` then builds no engine and no pipeline and shows "Weights REJECTED: ...". Adler-32 detects accidental corruption or a stale file; it is not a security hash.

### Chunking
A T=512 activation is 786,480 bytes, larger than a typical DataChannel message limit (262,144 in the Chromium run). `splitIntoChunks` cuts a finished packet into frames, each with a 20-byte header (`"CHNK"`, messageId, index, count, total) and no larger than the limit; `ChunkReassembler` rebuilds the exact original bytes, after which the normal packet checks run. The limit comes from `RTCPeerConnection.sctp.maxMessageSize` at send time. Only if the browser reports nothing usable does `main.js` use 65,536 (the RFC 8831 default); if it reports `Infinity`, the protocol's own 8 MiB reassembly bound is used.

### Frontend wiring (`Frontend/main.js`)
- `libs()` loads `mesh_pipeline.js` after the protocol and engine scripts (it reads both when it loads).
- `loadPartition(which)` downloads the manifest and the partition, verifies it, builds the engine, sets `S.fingerprint`, then calls `maybeInitPipeline()`.
- `dc.onopen` also calls `maybeInitPipeline()`. It creates `PipelineA` (host) or `PipelineB` (joiner) only when the partition is loaded **and** the channel is open, and only once.
- `dcTransport` is the `{send, maxMessageSize}` object the pipeline needs: waits for `open`, refuses frames above the negotiated limit, waits for `bufferedamountlow` while `bufferedAmount` > 1 MiB (resume at 256 KiB), and turns closure / `send()` exceptions into `CHANNEL_CLOSED` / `SEND_FAILED`.
- `onPacket` first lets `dc-test.js` claim its own test traffic, then queues (if the pipeline does not exist yet) or delivers each message to `pipe.handleMessage`, one at a time and in order. The old hand-written activation/token exchange is gone, so one packet is never processed by two handlers.
- If the channel closes while A is waiting, the pending request fails at once with `CHANNEL_CLOSED`.

### Known limits (not fixed here)
- Inference runs on the browser's main thread, so the page is unresponsive while a long prompt is computed. (Web Worker is future work.)
- The pipeline sends chunk frames of one packet in sequence, but two packets sent at the same moment could interleave their frames. Today only one request is in flight (A rejects a second with `BUSY`), so this does not occur.
- If B rejects its weights or never loads them, A reports "no HELLO received from Device B" after its timeout (20 s) rather than immediately.
- Only Chromium (headless, one machine) has been tested. Safari, Firefox and phones have not.


## 6. Stage 6 — failure handling and cleanup (as implemented)

- **One connection state** (`Frontend/connection_state.js`): `DISCONNECTED -> CONNECTING -> CONNECTED -> INFERENCE_READY`, plus `RECONNECTING` and `FAILED`. Events with no arrow from the current state are refused, so late or duplicate callbacks cannot corrupt state. `main.js` keeps `conn` as the single source of truth; the older `S.live`/`stage()` remain display details.
- **Connection generations** (`S.gen` in `main.js`): bumped by `closePeer()`. Every WebRTC/DataChannel callback and every pipeline transport (`makeTransport(gen)`) captures its generation and does nothing once it is stale. This is what stops an old session from touching a new one.
- **Timers** all go through `later()`/`cancel()`/`clearTimers()`, so cleanup can cancel every one of them.
- **Cleanup** is `closePeer()` (peer connection, DataChannel, pipeline, timers, ICE buffer) and `teardown()` (also the Socket.IO socket). Both are idempotent.
- **Recovery** is bounded: Device A re-offers up to `CONFIG.maxReconnects` (2) times with 1 s then 2 s backoff, each attempt limited to `reconnectEstablishMs`; Device B waits `reconnectWaitMs` per attempt. After that: `FAILED` with the reason. The budget refills only after a full successful inference round trip. Reconnection creates a brand-new peer connection and pipelines (no ICE restart).
- **Signalling**: `io({ reconnection: false })` (a silent Socket.IO reconnect would leave a socket that is in no room). `room-server` now emits `peer-left` and closes the room when a member leaves. Losing signalling after the DataChannel is open does not stop inference, but makes reconnection impossible.
- **Pipelines** (`engine/mesh_pipeline.js`): public idempotent `dispose(reason)` rejects the request in flight and every waiter immediately, clears timers and ignores later messages. An incomplete chunked message is discarded after `reassemblyTimeoutMs` of silence. `ChunkReassembler` ignores the remaining frames of a message it already rejected, so one broken transfer yields one ERROR packet instead of one per frame. The Stage 5 ERROR packet and validation are reused unchanged.

## 7. Stage 7 — performance measurement and the one engine change

**Request path with measured cost** (medians, Node loopback, optimized engine, T=35): A compute 161 ms → `P.encodeActivation` 0.31 ms → `P.splitIntoChunks` 0.01 ms (no split below the channel limit) → transport → `ChunkReassembler.push` (0 ms when one frame) → `P.decodePacket` 0.27 ms (includes the checksum) → `P.validateActivation` 0.02 ms → B compute 168 ms → reply. Everything except the two computes is 0.1–0.9% of a request. In the browser over a real DataChannel the A→B transfer of the 53,808-byte packet took 3.25 ms (median, same-page loopback).

**Cost model.** Local latency = A + B. Distributed latency for one request = A + serialize + transfer + B + reply, strictly sequential, so it cannot beat local. The benefit of splitting is memory per device (21.43 MB and 20.69 MB of weights instead of 42.12 MB).

**Measurement architecture (`benchmarks/`).** The benchmark code does not edit the pipeline. It temporarily wraps `P.encodeActivation`, `P.splitIntoChunks`, `P.decodePacket`, `P.validateActivation`, `P.ChunkReassembler.prototype.push`, `engine.forward` and `transport.send` (the pipeline looks these up on the protocol module at call time), then restores them. The `LoopbackLink` replaces the Stage 5 test link, which has a fixed 1 ms delay. Browser benchmarks use two `RTCPeerConnection`s in one page so both ends share a clock.

**The one source change.** `matmulTransposeB` in `engine/inference_engine.js` computes a 2×4 tile per pass instead of one output per pass. Each output still sums bias then `k = 0..K-1` in order in double precision, so results are bit-identical; the model, weights, precision, protocol and transport are unchanged. Measured about 2.0–2.4× faster. A second candidate (attention loop interchange) was reverted because no end-to-end gain could be demonstrated.

**Known performance limits (unchanged):** no KV cache (each token recomputes the whole sequence), attention is O(T²), compute runs on the main thread (UI freezes), and no real-network measurements exist yet.
