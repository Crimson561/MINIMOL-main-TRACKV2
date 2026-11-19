# Contributing to MINIMOL

MINIMOL is a learning-oriented portfolio project, but issues and pull requests are welcome. The project values **correctness and honest claims** over feature count.

## Set up

Follow [INSTALLATION.md](INSTALLATION.md) (Node, Python + PyTorch, `python model/export_weights.py`, `cd room-server && npm install`).

## Before opening a pull request

Run everything in [TESTING.md](TESTING.md#3-run-everything). All suites must pass; do not delete or weaken a test to get a green run.

```bash
python model/verify_reference.py
for t in test_tensor test_protocol test_js_engine test_distributed_inference test_stage5_pipeline test_stage6_reliability test_stage7_benchmarks; do node tests/$t.js || break; done
(cd room-server && npm test)
# optional, need Playwright + Chromium (TESTING.md section 5):
node tests/browser/ui_user_flow.js && node tests/browser/stage6_two_page.js
```

## Ground rules

- **Wire format.** Changing packet layout, magic bytes, message types or the fingerprint is a protocol change: bump `PROTOCOL_VERSION`, update `ARCHITECTURE.md` §5, and add tests. The `MESH` identifiers are historical and intentionally unchanged.
- **Numerics.** The JS engine must keep matching the PyTorch fixture (tolerance 1e-5), and distributed output must stay bit-identical to local. If you change the maths, regenerate the fixture with `python model/export_stage5_reference.py` and explain why.
- **Benchmarks.** Never edit files in `benchmarks/results/` by hand. New results come from the harness (`benchmarks/run.js`) and must state the machine and evidence level. Do not present same-machine numbers as network or device results.
- **Docs must match the code.** If you change behaviour, update the relevant doc in the same pull request. Label anything untested as untested.
- **No generated or private files.** Weights (`room-server/public/weights/`), `node_modules/` and profiler output are git-ignored. Never commit secrets, local absolute paths or personal data.
- **Scope.** Large changes (KV cache, WebGPU/WASM, TURN, new model) should start as an issue describing the design and how it will be verified.
- **Style.** Match the surrounding code. LF line endings (`.gitattributes` enforces this). Keep new dependencies to a minimum and say why they are needed.

## Reporting bugs

Include: what you did, what you expected, what happened, browser and OS, whether both devices were on one machine, and the console output of both pages. For connection problems, also state the network type (same Wi-Fi, mobile data, VPN).
