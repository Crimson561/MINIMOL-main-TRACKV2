"""
Model Weight Exporter for MESH Distributed AI Inference
Exports deterministically initialized weights (seed 42) into:
- partition_a.bin: Raw FP32 binary buffer for Phone A (wte, wpe, layers 0..2)
- partition_b.bin: Raw FP32 binary buffer for Phone B (layers 3..5, ln_f, lm_head)
- full_model.bin: Raw FP32 binary buffer for single-node / reference
- weights_manifest.json: Offsets, shapes, and metadata for fast browser typed array views.
"""

import sys
import os
import json
import struct
import torch
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.transformer import (
    MicroTransformer,
    init_weights_deterministic,
    SEED,
    EXPECTED_PARAMS,
    EXPECTED_FP32_BYTES,
)

EXPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "room-server", "public", "weights"))


def export_weights():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    print(f"Exporting deterministic model weights (seed {SEED}) to: {EXPORT_DIR}")

    model = MicroTransformer()
    init_weights_deterministic(model, seed=SEED)
    model.eval()

    state_dict = model.state_dict()

    # Define partition tensor assignments
    partition_a_keys = ["wte.weight", "wpe.weight"]
    for l in range(3):
        partition_a_keys.extend([
            f"h.{l}.ln_1.weight",
            f"h.{l}.ln_1.bias",
            f"h.{l}.attn.c_attn.weight",
            f"h.{l}.attn.c_attn.bias",
            f"h.{l}.attn.c_proj.weight",
            f"h.{l}.attn.c_proj.bias",
            f"h.{l}.ln_2.weight",
            f"h.{l}.ln_2.bias",
            f"h.{l}.mlp.c_fc.weight",
            f"h.{l}.mlp.c_fc.bias",
            f"h.{l}.mlp.c_proj.weight",
            f"h.{l}.mlp.c_proj.bias",
        ])

    partition_b_keys = []
    for l in range(3, 6):
        partition_b_keys.extend([
            f"h.{l}.ln_1.weight",
            f"h.{l}.ln_1.bias",
            f"h.{l}.attn.c_attn.weight",
            f"h.{l}.attn.c_attn.bias",
            f"h.{l}.attn.c_proj.weight",
            f"h.{l}.attn.c_proj.bias",
            f"h.{l}.ln_2.weight",
            f"h.{l}.ln_2.bias",
            f"h.{l}.mlp.c_fc.weight",
            f"h.{l}.mlp.c_fc.bias",
            f"h.{l}.mlp.c_proj.weight",
            f"h.{l}.mlp.c_proj.bias",
        ])
    partition_b_keys.extend(["ln_f.weight", "ln_f.bias", "lm_head.weight"])

    # Get parameters only (excluding non-trainable buffers like attn.bias mask)
    param_dict = dict(model.named_parameters())

    all_keys = list(param_dict.keys())
    assert set(partition_a_keys + partition_b_keys) == set(all_keys), f"Mismatch: {set(all_keys) ^ set(partition_a_keys + partition_b_keys)}"

    manifest = {
        "architecture": "GPT-style causal decoder",
        "vocab_size": 256,
        "context_length": 512,
        "num_layers": 6,
        "d_model": 384,
        "num_heads": 6,
        "d_head": 64,
        "d_ff": 1536,
        "seed": SEED,
        "total_parameters": EXPECTED_PARAMS,
        "total_fp32_bytes": EXPECTED_FP32_BYTES,
        "tensors": {},
        "partition_a": {"file": "partition_a.bin", "total_bytes": 0, "parameters": 0, "tensors": {}},
        "partition_b": {"file": "partition_b.bin", "total_bytes": 0, "parameters": 0, "tensors": {}},
        "full_model": {"file": "full_model.bin", "total_bytes": 0, "parameters": 0, "tensors": {}},
    }

    # Helper to pack binary
    def pack_tensors(keys, bin_filename):
        bin_path = os.path.join(EXPORT_DIR, bin_filename)
        offset = 0
        param_count = 0
        tensor_info = {}

        with open(bin_path, "wb") as f:
            for key in keys:
                t = param_dict[key].detach().cpu().numpy().astype(np.float32)
                raw_bytes = t.tobytes()
                num_elements = t.size
                shape = list(t.shape)
                byte_len = len(raw_bytes)

                f.write(raw_bytes)

                tensor_info[key] = {
                    "shape": shape,
                    "num_elements": num_elements,
                    "byte_offset": offset,
                    "byte_length": byte_len,
                    "dtype": "float32"
                }
                offset += byte_len
                param_count += num_elements

        return bin_path, offset, param_count, tensor_info

    # Pack Partition A
    pa_path, pa_bytes, pa_params, pa_tensors = pack_tensors(partition_a_keys, "partition_a.bin")
    manifest["partition_a"]["total_bytes"] = pa_bytes
    manifest["partition_a"]["parameters"] = pa_params
    manifest["partition_a"]["tensors"] = pa_tensors
    print(f"Partition A: {pa_params:,} params, {pa_bytes:,} bytes (~{pa_bytes / (1024 * 1024):.2f} MB)")

    # Pack Partition B
    pb_path, pb_bytes, pb_params, pb_tensors = pack_tensors(partition_b_keys, "partition_b.bin")
    manifest["partition_b"]["total_bytes"] = pb_bytes
    manifest["partition_b"]["parameters"] = pb_params
    manifest["partition_b"]["tensors"] = pb_tensors
    print(f"Partition B: {pb_params:,} params, {pb_bytes:,} bytes (~{pb_bytes / (1024 * 1024):.2f} MB)")

    # Pack Full Model
    full_path, full_bytes, full_params, full_tensors = pack_tensors(all_keys, "full_model.bin")
    manifest["full_model"]["total_bytes"] = full_bytes
    manifest["full_model"]["parameters"] = full_params
    manifest["full_model"]["tensors"] = full_tensors
    print(f"Full Model:  {full_params:,} params, {full_bytes:,} bytes (~{full_bytes / (1024 * 1024):.2f} MB)")

    # Global tensors dict
    manifest["tensors"] = full_tensors

    manifest_path = os.path.join(EXPORT_DIR, "weights_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to: {manifest_path}")
    print("[OK] All weights exported successfully.")


if __name__ == "__main__":
    export_weights()
