# MESH — Comprehensive Testing & Verification Guide

This document details the test progression and results for verifying the distributed inference system.

---

## Testing Progression

```
+----------------------------------------------------------------------------------+
| TEST A: Full Model Reference (PyTorch)                                           |
| -> Validates 11,040,768 params, seed 42, deterministic forward pass              |
+----------------------------------------------------------------------------------+
                                        |
                                        v
+----------------------------------------------------------------------------------+
| TEST B: Partition Correctness (PyTorch)                                          |
| -> Validates Full Model == Partition A (0-2) + Partition B (3-5) (Bit-Identical)  |
+----------------------------------------------------------------------------------+
                                        |
                                        v
+----------------------------------------------------------------------------------+
| TEST C: Communication & Binary Protocol (Node.js)                                |
| -> Validates framing, Adler-32 checksums, shape integrity, corruption rejection   |
+----------------------------------------------------------------------------------+
                                        |
                                        v
+----------------------------------------------------------------------------------+
| TEST D: Distributed Inference Engine (Browser / Node.js)                         |
| -> Validates JS engine executes partitions and matches PyTorch step-for-step     |
+----------------------------------------------------------------------------------+
                                        |
                                        v
+----------------------------------------------------------------------------------+
| TEST E: Physical Device Execution & Signalling Tests (Mobile Safari & Chrome)    |
| -> Live room connection, WebRTC STUN handshake, peer-to-peer generation          |
+----------------------------------------------------------------------------------+
```

---

## 1. Test A & B — Reference Model & Partition Equivalence

### Command:
```bash
python model/verify_reference.py
```

### Verified Results:
- **Parameter Count**:
  - `wte`: 98,304
  - `wpe`: 196,608
  - `6 x blocks`: 10,646,784
  - `ln_f`: 768
  - `lm_head`: 98,304
  - **Total**: **11,040,768** (FP32: 44,163,072 bytes)
- **Logit Difference (Full vs Partitioned)**: `0.00000000e+00` (Bit-identical)
- **PyTorch vs NumPy Reference Difference**: `1.07957480e-06` (Float32 epsilon)
- **10-Step Deterministic Generation on prompt `"MESH"`**:
  - Step 1: byte 228 (`0xe4`)
  - Step 2: byte 228 (`0xe4`)
  - Step 3: byte 228 (`0xe4`)
  - Step 4: byte 228 (`0xe4`)
  - Step 5: byte 228 (`0xe4`)
  - Step 6: byte 228 (`0xe4`)
  - Step 7: byte 98 (`'b'`)
  - Step 8: byte 98 (`'b'`)
  - Step 9: byte 240 (`0xf0`)
  - Step 10: byte 98 (`'b'`)
  - Sequence: `[77, 69, 83, 72, 228, 228, 228, 228, 228, 228, 98, 98, 240, 98]`

---

## 2. Test C — Communication Correctness & Protocol Validation

### Command:
```bash
node tests/test_protocol.js
```

### Verified Results:
1. **Activation Serialization**:
   - Shape `[1, 31, 384]` (47,616 bytes) packed with 48-byte header into 47,664 bytes.
   - Zero-copy Float32Array extraction verified bit-identical (`maxDiff = 0.0`).
2. **Adler-32 Checksum Validation**:
   - Bit flipped in payload: packet rejected with `Packet checksum mismatch`.
3. **Packet Header Boundaries**:
   - Truncated buffers (< 48 bytes) and corrupted magic headers rejected immediately.
4. **Token Serialization**:
   - Token payload correctly decoded with execution metrics.

---

## 3. Test D — Distributed Inference Simulation

### Command:
```bash
node tests/test_distributed_inference.js
```

### Verified Results:
Two independent client instances (Phone A holding only Partition A, Phone B holding only Partition B) connected over an asynchronous mock DataChannel:
- Step 1: Token `228` (Round-Trip: 69.0ms)
- Step 2: Token `228` (Round-Trip: 61.5ms)
- Step 3: Token `228` (Round-Trip: 61.0ms)
- Step 4: Token `228` (Round-Trip: 61.1ms)
- Step 5: Token `228` (Round-Trip: 77.3ms)
- Step 6: Token `228` (Round-Trip: 94.1ms)
- Step 7: Token `98` (Round-Trip: 93.3ms)
- Step 8: Token `98` (Round-Trip: 93.5ms)
- Step 9: Token `240` (Round-Trip: 93.8ms)
- Step 10: Token `98` (Round-Trip: 107.2ms)

Final Sequence:
`[77, 69, 83, 72, 228, 228, 228, 228, 228, 228, 98, 98, 240, 98]`
**Matches the PyTorch reference model bit-for-bit across all 10 tokens!**

---

## 4. Test E — Signalling Server Test Suite

### Command:
```bash
cd room-server && npm test
```

### Verified Results:
**40 of 40 tests passed:**
- 12/12 `roomManager.test.js` tests passed.
- 28/28 `signaling.test.js` tests passed:
  - Client creates room (ack + room-created event)
  - Second client joins with room code; both sides notified
  - Third client rejected with ROOM_FULL
  - Invalid / unknown room codes rejected
  - Peer disconnect / host promotion
  - WebRTC offer / answer / ICE candidate forwarding to room members only
  - Signalling payloads sanitized & validated
  - Rate limiting & per-IP connection limits enforced
  - All existing REST API endpoints preserved.
