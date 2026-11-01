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
