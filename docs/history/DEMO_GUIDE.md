# MESH — Live Demonstration Guide

Follow this guide to demonstrate real distributed AI inference across two smartphones or browser tabs.

---

## Demonstration Setup

### Step 1: Start the MESH Server

Open a terminal on your host computer (laptop/desktop):

```bash
cd room-server
npm start
```

Output:
```
[info] Server running at http://localhost:3000
```

### Step 2: Determine your LAN IP Address
- **Windows**: Run `ipconfig` (look for `IPv4 Address`, e.g. `192.168.1.15`).
- **macOS / Linux**: Run `ifconfig` or `ip a` (e.g. `192.168.1.15`).

Ensure both test devices (e.g., iPhone and Android phone, or laptop and phone) are connected to the **same local Wi-Fi network**.

---

## Demonstration Walkthrough

### 1. Connecting Device 1 (Phone A — Host)
1. On Device 1 (e.g. Android Chrome or Laptop Tab 1), navigate to:
   ```
   http://<YOUR_LAN_IP>:3000
   ```
2. The page loads with status pills:
   - `Socket.IO`: **Connected**
   - `Room`: **None**
   - `WebRTC P2P`: **Idle**
   - `Model Weights`: **Not Loaded**
3. Tap **"Create Room (Host / Phone A)"**.
4. The system:
   - Emits `create-room` to the signalling server.
   - Assigns Device 1 as **PHONE A (Host)**.
   - Displays a unique room code, e.g. **`MESH-H8Y2P`**.
   - Generates a shareable URL: `http://<YOUR_LAN_IP>:3000?room=MESH-H8Y2P`.
   - Downloads `partition_a.bin` (21.43 MB, layers 0–2).
   - Initializes `MeshPartitionA` in RAM.
   - Status updates: `Model Weights: Ready (21.4MB)`.

---

### 2. Connecting Device 2 (Phone B — Peer)
1. On Device 2 (e.g. iPhone Safari or Laptop Tab 2), navigate to:
   ```
   http://<YOUR_LAN_IP>:3000
   ```
   *(Or scan / click the share link)*.
2. In the room code box, enter the room code (e.g. `MESH-H8Y2P`) and tap **"Join Room (Peer / Phone B)"**.
3. The system:
   - Assigns Device 2 as **PHONE B (Peer)**.
   - Downloads `partition_b.bin` (20.69 MB, layers 3–5 + head).
   - Initializes `MeshPartitionB` in RAM.
   - Status updates: `Model Weights: Ready (20.7MB)`.

---

### 3. Automatic WebRTC P2P Handshake
1. As soon as Device 2 joins, the signalling server sends `peer-joined` to Device 1.
2. Device 1 creates an `RTCPeerConnection` with a binary `DataChannel` named `"mesh"`.
3. Device 1 sends the SDP offer; Device 2 responds with the SDP answer.
4. ICE candidates are exchanged over STUN (`stun:stun.l.google.com:19302`).
5. Connection establishes directly between the devices:
   - `WebRTC P2P`: **connected**
   - `DataChannel`: **Open (Binary)**
6. The Signalling server is now silent — inference will travel directly between the two phones.

---

### 4. Running Distributed Inference
1. On Device 1, the input prompt box and **"Start Inference"** button become enabled.
2. Enter a prompt (e.g. `MESH`) and token count (e.g. `10`).
3. Tap **"Start Inference"**.
4. Observe the live generation pipeline:
   - **Step 1**:
     - Phone 1 runs Embedding + Layers 0, 1, 2 on local CPU (~20-30 ms).
     - Phone 1 serializes the intermediate activation tensor `[1, 4, 384]` (6.0 KB).
     - Phone 1 sends the binary packet directly across the DataChannel to Phone 2.
     - Phone 2 receives and validates the activation tensor.
     - Phone 2 executes Layers 3, 4, 5 + LM Head on local CPU (~20-30 ms).
     - Phone 2 selects the next byte token via greedy decoding (argmax).
     - Phone 2 sends the selected byte token back across the DataChannel to Phone 1.
     - Phone 1 appends the token to its context.
   - **Steps 2 to 10**:
     - The process repeats autoregressively until 10 tokens are generated.
5. Watch the dashboard table update with **real measurements**:
   - Phone A Compute Time
   - Phone B Compute Time
   - Wire Payload Size
   - Round-Trip Latency per Token
   - Throughput (tokens/sec)
   - Total Bytes Transferred over DataChannel.

---

## Two-Tab Desktop Simulation

If two physical phones are not currently available:
1. Open two browser windows side by side (e.g., Chrome and Firefox, or two Chrome tabs).
2. Follow the exact sequence above: Window 1 creates the room; Window 2 joins with the room code.
3. Observe direct WebRTC communication between the two tabs!
