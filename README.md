# MESH — Peer-to-Peer Distributed AI Inference

[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

**MESH** is a peer-to-peer distributed inference platform that splits a genuine **Byte-Level Micro-Transformer** across two independent devices (such as mobile smartphones or browser clients) communicating directly over a **WebRTC DataChannel**.

Neither device runs the entire model. Both devices actively compute complementary Transformer layers for every single output token generated, with zero intermediate tensor relay through the central server.

---

## 1. System Overview

```
                      +-----------------------------+
                      |   Node.js Signalling Server |
                      |    (Room Code & SDP/ICE)    |
                      +--------------+--------------+
                                     |
               Signalling Handshake Only (No Activation Relays)
                                     |
    +--------------------------------+--------------------------------+
    |                                                                 |
    v                                                                 v
+-----------------------+                         +-----------------------+
|        PHONE A        |    WebRTC DataChannel   |        PHONE B        |
|  (Host / Partition A) |========================>|  (Peer / Partition B) |
|                       |   Binary Activations    |                       |
| • Prompt UTF-8 Bytes  |   Shape: [1, T, 384]    | • Receives Act [1,T,384]
| • Byte Embedding(wte) |                         | • Layer 3             |
| • Pos Embedding(wpe)  |   WebRTC DataChannel    | • Layer 4             |
| • Layer 0             |<========================| • Layer 5             |
| • Layer 1             |      Selected Token     | • Final LayerNorm ln_f|
| • Layer 2             |       Byte (0..255)     | • Output Head lm_head |
| • Sends Activations   |                         | • Greedy Argmax Token |
+-----------------------+                         +-----------------------+
```

---

## 2. Model Architecture & Exact Parameter Verification

The architecture is a GPT-style causal decoder operating directly on raw bytes (vocabulary size 256):

| Hyperparameter | Value | Description |
|---|---|---|
| **Vocabulary** | 256 | Raw byte tokens (0..255) |
| **Context Length** | 512 | Maximum context sequence length |
| **Transformer Layers** | 6 | Total causal attention blocks |
| **Model Dimension ($D$)** | 384 | Hidden dimension |
| **Attention Heads ($H$)** | 6 | 64 dimensions per head |
| **Feed-Forward Dimension ($D_{ff}$)** | 1536 | $4 \times D$ expansion |
| **Random Seed** | 42 | Deterministic weight initialization |
| **Temperature** | 0.0 | Deterministic greedy decoding |
| **Phone A Allocation** | Layers 0–2 | Embedding + Positional + 3 Blocks |
| **Phone B Allocation** | Layers 3–5 | 3 Blocks + Final Norm + LM Head |
| **Execution Environment** | Browser CPU | Float32Array typed arrays (No WebGPU/WebGL) |
| **Communication** | Direct WebRTC | P2P Binary DataChannel |

### Exact Parameter Count Verification: **11,040,768**

The exact mathematical breakdown confirms the specification to the individual parameter:

1. **Byte Embedding (`wte`)**: $256 \times 384 = 98,304$
2. **Positional Embedding (`wpe`)**: $512 \times 384 = 196,608$
3. **Each Transformer Block**:
   - `ln_1` (weight + bias): $384 + 384 = 768$
   - `attn.c_attn` (Q,K,V weights + biases): $(384 \times 1152) + 1152 = 443,520$
   - `attn.c_proj` (weight + bias): $(384 \times 384) + 384 = 147,840$
   - `ln_2` (weight + bias): $384 + 384 = 768$
   - `mlp.c_fc` (weight + bias): $(384 \times 1536) + 1536 = 591,360$
   - `mlp.c_proj` (weight + bias): $(1536 \times 384) + 384 = 590,208$
   - **Total per block**: $1,774,464$ parameters
4. **All 6 Blocks**: $6 \times 1,774,464 = 10,646,784$ parameters
5. **Final LayerNorm (`ln_f`)**: $384 + 384 = 768$ parameters
6. **LM Output Head (`lm_head`)**: $256 \times 384 = 98,304$ parameters (untied weights, no bias)

$$\mathbf{Grand\ Total} = 98,304 + 196,608 + 10,646,784 + 768 + 98,304 = \mathbf{11,040,768\ parameters}$$

In FP32 (4 bytes per parameter):
$$11,040,768 \times 4\text{ bytes} = 44,163,072\text{ bytes} \approx \mathbf{44.16\ MB}$$

### Distributed Partition Sizes:
- **Phone A (Partition A)**: $5,618,304\text{ params}$ ($22,473,216\text{ bytes} \approx 21.43\text{ MB}$)
- **Phone B (Partition B)**: $5,422,464\text{ params}$ ($21,689,856\text{ bytes} \approx 20.69\text{ MB}$)
Each phone only needs to download its own half of the weights!

---

## 3. Quick Start & Execution

### Prerequisites
- Node.js >= 18
- Python >= 3.10 with PyTorch and NumPy (for reference verification)

### Installation
```bash
# 1. Install signaling server dependencies
cd room-server
npm install
cd ..

# 2. Run reference verification and export deterministic weights
python model/verify_reference.py
python model/export_weights.py
```

### Running All Automated Correctness Tests
```bash
# Test A & B: PyTorch reference & Partition bit-exact verification
python model/verify_reference.py

# Test C: WebRTC Binary Protocol & DataChannel framing verification
node tests/test_protocol.js

# Test D: Distributed inference end-to-end simulation
node tests/test_distributed_inference.js

# Test Signalling Server & Room Manager test suite (40 tests)
cd room-server && npm test && cd ..

# Performance Benchmarking Suite
node tests/benchmark.js
```

### Starting the Live System
```bash
cd room-server
npm start
```
The server will boot at `http://localhost:3000` (or your local IP address `http://<LAN_IP>:3000`).

---

## 4. Physical Smartphone Demonstration Sequence

1. Open your laptop / terminal and run:
   ```bash
   cd room-server
   npm start
   ```
2. Note your computer's local IP address (e.g., `http://192.168.1.50:3000`). Ensure both phones are connected to the same Wi-Fi network.
3. **On Phone 1 (Android Chrome)**:
   - Navigate to `http://192.168.1.50:3000`.
   - Tap **"Create Room (Host / Phone A)"**.
   - A room code appears (e.g., `MESH-AB3K9`). Phone 1 loads Partition A weights (~21.4 MB).
4. **On Phone 2 (iOS Safari or Android Chrome)**:
   - Navigate to `http://192.168.1.50:3000`.
   - Enter the room code or tap the share link.
   - Tap **"Join Room (Peer / Phone B)"**. Phone 2 loads Partition B weights (~20.7 MB).
5. The WebRTC handshake completes automatically over STUN (`stun:stun.l.google.com:19302`), and the **DataChannel Status turns to "Open (Binary)"**.
6. **On Phone 1**:
   - Enter a prompt (e.g. `MESH`).
   - Tap **"Start Inference"**.
7. Observe live peer-to-peer generation:
   - Phone 1 computes layers 0–2.
   - Activations are sent directly over WebRTC to Phone 2.
   - Phone 2 computes layers 3–5 and LM Head.
   - Phone 2 selects the next byte token and sends it back to Phone 1.
   - Both screens update live with real byte values and performance statistics!
