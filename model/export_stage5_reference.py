"""
Stage 5 reference data. Runs the PyTorch reference model (seed 42) on deterministic inputs and records,
for each one, the intermediate activation after layer 2 (what Device A must produce and transfer) and the
final last-position logits (what Device B must produce). Tests and the browser check compare against this.

Run AFTER model/export_weights.py (it copies the partition checksums from the manifest):
    python model/export_weights.py
    python model/export_stage5_reference.py
"""
import sys, os, json, base64, numpy as np, torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from model.transformer import MicroTransformer, init_weights_deterministic, SEED

MANIFEST = os.path.join(ROOT, "room-server", "public", "weights", "weights_manifest.json")
OUT = os.path.join(ROOT, "tests", "fixtures", "stage5_reference.json")

TEXT = "The quick brown fox jumps over the lazy dog. "
CASES = [
    ("MESH", list("MESH".encode())),
    ("sentence31", list("MESH distributed inference test".encode())),
    ("hello13", list("Hello, world!".encode())),
    ("single1", list("a".encode())),
    ("repeat135", list((TEXT * 3).encode())),
    # 200 arbitrary bytes (incl. non-ASCII). 200 * 1536 B = 307 KB: bigger than a 256 KiB DataChannel message.
    ("long200", [(i * 37 + 11) % 256 for i in range(200)]),
]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def main():
    if not os.path.exists(MANIFEST):
        sys.exit("weights_manifest.json not found. Run model/export_weights.py first.")
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    model = MicroTransformer()
    init_weights_deterministic(model, seed=SEED)
    model.eval()
    cases = []
    with torch.no_grad():
        for name, ids in CASES:
            idx = torch.tensor([ids], dtype=torch.long)
            x = model.wte(idx) + model.wpe(torch.arange(len(ids)).unsqueeze(0))
            for blk in model.h[:3]:
                x = blk(x)
            act = x.clone()                       # [1, T, 384] after layer 2
            for blk in model.h[3:]:
                x = blk(x)
            logits = model.lm_head(model.ln_f(x))[0, -1]
            cases.append({"name": name, "ids": ids, "T": len(ids), "argmax": int(logits.argmax()),
                          "act_shape": list(act.shape), "act_b64": b64(act.numpy()), "logits_b64": b64(logits.numpy())})
            print(f"{name:11s} T={len(ids):3d} argmax={cases[-1]['argmax']:3d}")
    out = {"seed": SEED, "torch_version": torch.__version__, "tolerance": 1e-5, "d_model": 384, "vocab": 256,
           "partition_a_adler32": manifest["partition_a"]["checksum_adler32"],
           "partition_b_adler32": manifest["partition_b"]["checksum_adler32"], "cases": cases}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"))
    print(f"Wrote {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
