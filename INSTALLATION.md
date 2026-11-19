# MINIMOL — Installation

Everything here was run during the Stage 8 audit on **Linux x86-64, Node.js 22.23, npm 10.9, Python 3.12.3, torch 2.15.0, numpy 2.5.4**, starting from a clean copy of the repository's files. Other versions and operating systems are untested. Paths below use `/`; on Windows use `\` or Git Bash.

## 1. Prerequisites

| Need | Why | Notes |
|---|---|---|
| **Node.js** (22 tested; 18+ expected to work) and npm | Signalling server, all JavaScript tests | https://nodejs.org |
| **Python 3** (3.12 tested) | One-off weight generation and the PyTorch reference check | |
| **PyTorch + NumPy** | `export_weights.py` imports `torch` | See the size warning below |
| **A Chromium-based browser** (Chrome, Edge) | The app. Only headless Chromium was exercised in the project's own browser tests | Firefox/Safari untested |
| ~250 MB free disk | Weights (~88 MB) + dependencies | PyTorch itself needs more |

> **PyTorch is large.** On Linux, `pip install torch` pulls CUDA libraries (several GB). If you have no NVIDIA GPU, install the CPU build instead, as PyTorch's own instructions describe: `pip install torch --index-url https://download.pytorch.org/whl/cpu` (this exact command was not tested in the audit environment, which could not reach that index). Torch is needed **only** to generate the weights and run the Python checks, never to run the app.

## 2. Get the code and install dependencies

```bash
git clone <your-repository-url> minimol
cd minimol

# Python (weight export + reference checks)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Node (signalling server)
cd room-server
npm install
cd ..
```

The app itself has **no build step** and no bundler: the browser loads plain scripts served by the Node server.

## 3. Generate the model weights

The weights are **not stored in git** (they are ~88 MB and fully reproducible). From the repository root:

```bash
python model/export_weights.py
python model/export_stage5_reference.py     # optional: regenerates tests/fixtures/stage5_reference.json
```

`export_weights.py` builds the model with random seed 42, exports it, and writes to `room-server/public/weights/`:

```
partition_a.bin        22,473,216 bytes   (Device A: embeddings + blocks 0-2)
partition_b.bin        21,689,856 bytes   (Device B: blocks 3-5 + final norm + head)
full_model.bin         44,163,072 bytes   (reference / local runs)
weights_manifest.json  tensor table + Adler-32 checksums
```

Because the seed is fixed, every machine produces **identical files**. The audit confirmed this: regenerating the test fixture on a different machine produced a byte-identical file to the committed one.

Optional check that the Python reference and the split model agree:

```bash
python model/verify_reference.py     # expect: ALL REFERENCE VERIFICATION TESTS PASSED SUCCESSFULLY!
```

> The model is **randomly initialised and untrained**. Everything works, but the generated text is meaningless. There is no training script in this repository.

## 4. Start the server

```bash
cd room-server
npm start               # same as: node server.js
```

Expected output: `MINIMOL room server listening on http://0.0.0.0:3000`. To use another port: `PORT=3100 npm start` (Windows PowerShell: `$env:PORT=3100; npm start`).

The server listens on all interfaces, serves the frontend at `/`, the engine and protocol scripts, and the weights at `/weights/`.

## 5. Open the app

- **Same computer, two tabs:** open `http://localhost:3000` in two tabs. This is the easiest way to try everything.
- **Two devices on one network:** find the host computer's LAN address (`ip addr`, `ifconfig` or `ipconfig`) and open `http://<LAN-IP>:3000` on both. Allow inbound TCP on the server port in the host firewall. This path is **not yet verified on physical devices**; see [TESTING.md](TESTING.md#7-physical-device-checklist-not-yet-run).

Then follow the usage steps in the [README](README.md#usage).

### Network and WebRTC requirements

- **STUN only, no TURN.** The configured ICE server is Google's public STUN server (`stun:stun.l.google.com:19302`). Two devices on the same LAN should connect directly; devices on different networks may connect through STUN if their NATs allow it. Devices behind symmetric NATs, many corporate or campus networks, and some mobile carriers need a **TURN relay**, which this project does not include. When no path exists the connection simply fails and the app reports `FAILED`.
- **Offline use.** The page loads fonts from Google Fonts; without internet it falls back to system fonts. A LAN-only setup should not need STUN, but this was not tested.
- **mDNS host candidates.** Chrome hides local IP addresses behind `.local` names. Same-LAN connections normally still work; if mDNS is blocked on your network, direct connection can fail. (The project's own two-page browser test disables this feature to connect pages on one machine.)
- **HTTPS.** Not required by anything MINIMOL uses (no camera/microphone). If a browser on a physical device refuses a feature on a plain-`http` address, serve over HTTPS. Details in [TESTING.md](TESTING.md#https-and-secure-contexts).

## 6. Verify the installation

```bash
node tests/test_tensor.js           # expect: TENSOR TESTS PASSED: 38 passed, 0 failed
(cd room-server && npm test)        # expect: SIGNALLING TESTS PASSED: 15 passed, 0 failed
```

The full test list is in [TESTING.md](TESTING.md).

## 7. Common errors

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | PyTorch not installed in the active environment | Activate the venv; `pip install -r requirements.txt` |
| `ValueError: libcublasLt.so... not found` after installing torch with `--no-deps` | CUDA wheel installed without its CUDA libraries | Install normally (`pip install torch`) or use the CPU build |
| `Weights missing. Run: python model/export_weights.py` | Weights not generated | Section 3 |
| `Error: Cannot find module 'express'` / `'socket.io'` | `npm install` not run in `room-server/` | `cd room-server && npm install` |
| `EADDRINUSE` (port in use) | Another process on 3000 | Stop it, or use `PORT=3100 npm start` |
| Page loads, weights fail to download (404) | Server started before weights existed, or from the wrong folder | Generate weights, restart with `npm start` inside `room-server/` |
| "Weights REJECTED" in the app | Browser cached a stale `.bin` from an older export | Hard-reload (Ctrl+Shift+R) or clear site data; regenerate weights if you edited the export |
| Other device cannot open `http://<LAN-IP>:3000` | Firewall, wrong network, client isolation on guest Wi-Fi | Allow the port; use the same non-isolated network |
| Devices pair but the connection ends in `FAILED` | No direct path (NAT/firewall) and no TURN | Same Wi-Fi; or add a TURN server to the `iceServers` list where `Frontend/main.js` creates the `RTCPeerConnection` (it currently lists only `CONFIG.stun`) |
| UI freezes while generating | Inference runs on the main thread (known limit) | Wait; use a short prompt |
| `peer-left` / room closes when a tab reloads | By design: a room closes when a member leaves | Create a new room |
