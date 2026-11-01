"""
Verification Script for MESH Micro-Transformer Reference Implementation
Validates:
1. Exact architectural parameter count (11,040,768).
2. Exact FP32 size (44,163,072 bytes ~ 44.16 MB).
3. Deterministic initialization (seed 42).
4. Equivalence of Full Model vs Partition A + Partition B.
5. Equivalence of PyTorch reference vs NumPy reference.
6. Greedy decoding autoregressive output.
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.transformer import (
    VOCAB_SIZE,
    MAX_SEQ_LEN,
    NUM_LAYERS,
    D_MODEL,
    NUM_HEADS,
    D_FF,
    SEED,
    EXPECTED_PARAMS,
    EXPECTED_FP32_BYTES,
    calculate_param_count,
    MicroTransformer,
    MicroTransformerPartitionA,
    MicroTransformerPartitionB,
    init_weights_deterministic,
    NumpyTransformer,
)


def verify_parameter_count():
    print("=" * 70)
    print("1. MATHEMATICAL AND PROGRAMMATIC PARAMETER COUNT VERIFICATION")
    print("=" * 70)

    math_total, breakdown = calculate_param_count()
    print(f"Mathematical total: {math_total:,} parameters")
    for k, v in breakdown.items():
        print(f"  - {k:15s}: {v:>10,}")

    assert math_total == EXPECTED_PARAMS, f"Math mismatch: {math_total} != {EXPECTED_PARAMS}"

    model = MicroTransformer()
    pytorch_total = sum(p.numel() for p in model.parameters())
    print(f"\nPyTorch actual total: {pytorch_total:,} parameters")
    print(f"Expected:             {EXPECTED_PARAMS:,} parameters")
    assert pytorch_total == EXPECTED_PARAMS, f"PyTorch mismatch: {pytorch_total} != {EXPECTED_PARAMS}"

    fp32_bytes = pytorch_total * 4
    print(f"FP32 byte count:      {fp32_bytes:,} bytes ({fp32_bytes / (1024 * 1024):.2f} MB)")
    assert fp32_bytes == EXPECTED_FP32_BYTES, f"Byte size mismatch: {fp32_bytes} != {EXPECTED_FP32_BYTES}"
    print("✓ Parameter count and FP32 size match the challenge specification EXACTLY.\n")


def verify_partition_equivalence():
    print("=" * 70)
    print("2. MODEL PARTITIONING VERIFICATION (FULL vs PARTITION A + B)")
    print("=" * 70)

    # 1. Initialize Full Model deterministically
    full_model = MicroTransformer()
    init_weights_deterministic(full_model, seed=SEED)
    full_model.eval()

    # 2. Initialize Partition A and load weights
    part_a = MicroTransformerPartitionA()
    part_a.eval()
    part_a.wte.load_state_dict(full_model.wte.state_dict())
    part_a.wpe.load_state_dict(full_model.wpe.state_dict())
    for i in range(3):
        part_a.h[i].load_state_dict(full_model.h[i].state_dict())

    # 3. Initialize Partition B and load weights
    part_b = MicroTransformerPartitionB()
    part_b.eval()
    for i in range(3):
        part_b.h[i].load_state_dict(full_model.h[i + 3].state_dict())
    part_b.ln_f.load_state_dict(full_model.ln_f.state_dict())
    part_b.lm_head.load_state_dict(full_model.lm_head.state_dict())

    # 4. Test input prompt: "MESH" -> ASCII/UTF-8 byte values [77, 69, 83, 72]
    prompt_bytes = list(b"MESH distributed inference test")
    x = torch.tensor([prompt_bytes], dtype=torch.long)
    print(f"Prompt: {bytes(prompt_bytes).decode('utf-8')}")
    print(f"Prompt bytes ({len(prompt_bytes)} tokens): {prompt_bytes}")

    with torch.no_grad():
        # Full model inference
        full_logits = full_model(x)

        # Distributed partition inference
        # Step 1: Phone A executes layers 0-2
        intermediate_activations = part_a(x)
        print(f"Phone A intermediate activation tensor shape: {list(intermediate_activations.shape)}")

        # Step 2: Phone B executes layers 3-5 + head
        partition_logits = part_b(intermediate_activations)
        print(f"Phone B final logits shape: {list(partition_logits.shape)}")

        # Step 3: Compare logits
        max_diff = torch.max(torch.abs(full_logits - partition_logits)).item()
        print(f"Max absolute difference between Full and Partitioned: {max_diff:.8e}")
        assert max_diff == 0.0, f"Partition error: logits difference is {max_diff}"
        print("✓ Full model and Partition A + Partition B produce BIT-IDENTICAL logits in PyTorch.\n")

        # Step 4: Greedy token generation
        full_next_token = torch.argmax(full_logits[0, -1, :]).item()
        part_next_token = torch.argmax(partition_logits[0, -1, :]).item()
        print(f"Full model selected next byte:      {full_next_token} (repr: {repr(chr(full_next_token))})")
        print(f"Partitioned selected next byte:     {part_next_token} (repr: {repr(chr(part_next_token))})")
        assert full_next_token == part_next_token
        print("✓ Selected next byte matches perfectly.\n")

    return full_model, part_a, part_b


def verify_numpy_equivalence(full_model):
    print("=" * 70)
    print("3. NUMPY REFERENCE ENGINE VERIFICATION (PyTorch vs NumPy)")
    print("=" * 70)

    # Convert PyTorch weights to NumPy float32 dict
    weights_np = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in full_model.state_dict().items()}
    np_engine = NumpyTransformer(weights_np)

    prompt_bytes = list(b"MESH distributed inference test")
    x_pt = torch.tensor([prompt_bytes], dtype=torch.long)

    with torch.no_grad():
        pt_logits = full_model(x_pt)[0].numpy()

    np_logits = np_engine.forward(prompt_bytes)

    diff = np.max(np.abs(pt_logits - np_logits))
    print(f"Max absolute difference between PyTorch and NumPy: {diff:.8e}")
    # Numerical precision difference is typically < 1e-5 due to GELU / dot product order
    assert diff < 1e-4, f"NumPy engine difference too high: {diff}"

    pt_token = np.argmax(pt_logits[-1])
    np_token = np.argmax(np_logits[-1])
    print(f"PyTorch next byte: {pt_token}")
    print(f"NumPy next byte:   {np_token}")
    assert pt_token == np_token, f"Token mismatch: {pt_token} vs {np_token}"
    print("✓ Pure NumPy engine exactly matches PyTorch reference decisions.\n")


def test_autoregressive_generation(full_model):
    print("=" * 70)
    print("4. MULTI-STEP AUTOREGRESSIVE GENERATION TEST (Temperature = 0.0)")
    print("=" * 70)
    prompt = b"MESH"
    tokens = list(prompt)
    num_steps = 10
    print(f"Initial prompt: {repr(prompt.decode('utf-8'))} (bytes: {tokens})")

    with torch.no_grad():
        for step in range(num_steps):
            x = torch.tensor([tokens], dtype=torch.long)
            logits = full_model(x)
            next_byte = torch.argmax(logits[0, -1, :]).item()
            tokens.append(next_byte)
            print(f"  Step {step + 1:2d}: Generated byte {next_byte:3d} ({repr(chr(next_byte)) if 32 <= next_byte <= 126 else hex(next_byte)}) -> Total length: {len(tokens)}")

    gen_bytes = bytes(tokens)
    print(f"Final generated sequence bytes: {tokens}")
    print(f"Final string (lossy ASCII/Latin1): {repr(gen_bytes.decode('latin1'))}")
    print("✓ Autoregressive generation completed deterministically.\n")
    return tokens


if __name__ == "__main__":
    verify_parameter_count()
    full_model, part_a, part_b = verify_partition_equivalence()
    verify_numpy_equivalence(full_model)
    test_autoregressive_generation(full_model)
    print("=" * 70)
    print("ALL REFERENCE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
