"""
Byte-Level Micro-Transformer (MESH Core Reference Model)
Architecture: GPT-style causal decoder
Vocabulary: 256 byte values
Context length: 512
Layers: 6
Hidden dimension (d_model): 384
Attention heads: 6 (head dimension: 64)
Feed-forward dimension (d_ff): 1536
Deterministic seed: 42
Parameter count: 11,040,768 (FP32: 44,163,072 bytes ~ 44.16 MB)
"""

import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==============================================================================
# Model Configurations & Parameter Verification
# ==============================================================================
VOCAB_SIZE = 256
MAX_SEQ_LEN = 512
NUM_LAYERS = 6
D_MODEL = 384
NUM_HEADS = 6
D_HEAD = D_MODEL // NUM_HEADS  # 64
D_FF = 1536
LAYER_NORM_EPS = 1e-5
SEED = 42

EXPECTED_PARAMS = 11_040_768
EXPECTED_FP32_BYTES = EXPECTED_PARAMS * 4  # 44,163,072 bytes (~44.16 MB)


def calculate_param_count():
    """
    Mathematical breakdown of parameters:
    - Token Embedding: 256 * 384 = 98,304
    - Positional Embedding: 512 * 384 = 196,608
    - Per Transformer Block:
      - ln_1 (weight + bias): 384 + 384 = 768
      - attn c_attn (Q,K,V weights + biases): (384 * 1152) + 1152 = 443,520
      - attn c_proj (weight + bias): (384 * 384) + 384 = 147,840
      - ln_2 (weight + bias): 384 + 384 = 768
      - mlp c_fc (weight + bias): (384 * 1536) + 1536 = 591,360
      - mlp c_proj (weight + bias): (1536 * 384) + 384 = 590,208
      Total per block = 768 + 443,520 + 147,840 + 768 + 591,360 + 590,208 = 1,774,464
    - 6 Blocks = 6 * 1,774,464 = 10,646,784
    - ln_f (weight + bias): 384 + 384 = 768
    - lm_head (weight, no bias): 256 * 384 = 98,304
    Grand Total = 98,304 + 196,608 + 10,646,784 + 768 + 98,304 = 11,040,768
    """
    wte = VOCAB_SIZE * D_MODEL
    wpe = MAX_SEQ_LEN * D_MODEL
    block_params = (
        (D_MODEL * 2) +                      # ln_1 (weight, bias)
        (D_MODEL * 3 * D_MODEL + 3 * D_MODEL) + # attn c_attn (weight, bias)
        (D_MODEL * D_MODEL + D_MODEL) +       # attn c_proj (weight, bias)
        (D_MODEL * 2) +                      # ln_2 (weight, bias)
        (D_MODEL * D_FF + D_FF) +             # mlp c_fc (weight, bias)
        (D_FF * D_MODEL + D_MODEL)            # mlp c_proj (weight, bias)
    )
    all_blocks = NUM_LAYERS * block_params
    ln_f = D_MODEL * 2
    lm_head = VOCAB_SIZE * D_MODEL            # bias = False
    total = wte + wpe + all_blocks + ln_f + lm_head
    return total, {
        "wte": wte,
        "wpe": wpe,
        "block_single": block_params,
        "blocks_all": all_blocks,
        "ln_f": ln_f,
        "lm_head": lm_head,
        "total": total
    }


# ==============================================================================
# PyTorch Reference Implementation
# ==============================================================================
if HAS_TORCH:
    class CausalSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = nn.Linear(D_MODEL, 3 * D_MODEL, bias=True)
            self.c_proj = nn.Linear(D_MODEL, D_MODEL, bias=True)
            self.n_head = NUM_HEADS
            self.n_embd = D_MODEL

            # Causal mask buffer
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN)).view(
                    1, 1, MAX_SEQ_LEN, MAX_SEQ_LEN
                ),
            )

        def forward(self, x):
            B, T, C = x.size()
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)

            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

            # Scaled dot-product attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v  # (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.c_proj(y)


    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_fc = nn.Linear(D_MODEL, D_FF, bias=True)
            self.c_proj = nn.Linear(D_FF, D_MODEL, bias=True)

        def forward(self, x):
            x = self.c_fc(x)
            x = F.gelu(x, approximate="tanh")
            x = self.c_proj(x)
            return x


    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_1 = nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPS)
            self.attn = CausalSelfAttention()
            self.ln_2 = nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPS)
            self.mlp = MLP()

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


    class MicroTransformer(nn.Module):
        """
        Complete reference model: 6 layers, byte vocabulary.
        """
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(VOCAB_SIZE, D_MODEL)
            self.wpe = nn.Embedding(MAX_SEQ_LEN, D_MODEL)
            self.h = nn.ModuleList([Block() for _ in range(NUM_LAYERS)])
            self.ln_f = nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPS)
            self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

        def forward(self, idx):
            B, T = idx.size()
            assert T <= MAX_SEQ_LEN, f"Sequence length {T} exceeds max context {MAX_SEQ_LEN}"
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)

            tok_emb = self.wte(idx)
            pos_emb = self.wpe(pos)
            x = tok_emb + pos_emb

            for block in self.h:
                x = block(x)

            x = self.ln_f(x)
            logits = self.lm_head(x)
            return logits

        def generate_next_token(self, idx):
            """Deterministic greedy decoding (temperature=0.0)"""
            logits = self(idx)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            return next_token


    class MicroTransformerPartitionA(nn.Module):
        """
        Phone A Partition:
        - Byte embedding (wte)
        - Positional encoding (wpe)
        - Layers 0, 1, 2
        Outputs intermediate activations shape [B, T, 384].
        """
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(VOCAB_SIZE, D_MODEL)
            self.wpe = nn.Embedding(MAX_SEQ_LEN, D_MODEL)
            self.h = nn.ModuleList([Block() for _ in range(3)])  # layers 0, 1, 2

        def forward(self, idx):
            B, T = idx.size()
            assert T <= MAX_SEQ_LEN
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
            x = self.wte(idx) + self.wpe(pos)
            for block in self.h:
                x = block(x)
            return x  # Intermediate activations


    class MicroTransformerPartitionB(nn.Module):
        """
        Phone B Partition:
        - Layers 3, 4, 5
        - Final LayerNorm (ln_f)
        - LM Head (lm_head)
        Receives intermediate activations shape [B, T, 384], outputs logits [B, T, 256].
        """
        def __init__(self):
            super().__init__()
            self.h = nn.ModuleList([Block() for _ in range(3)])  # layers 3, 4, 5
            self.ln_f = nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPS)
            self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

        def forward(self, x):
            for block in self.h:
                x = block(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            return logits


def init_weights_deterministic(model, seed=SEED):
    """
    Deterministic initialization using specified seed (42).
    Linear and embedding weights ~ N(0, 0.02)
    Biases = 0.0
    LayerNorm weight = 1.0, bias = 0.0
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for init_weights_deterministic")

    torch.manual_seed(seed)
    for name, param in model.named_parameters():
        if "ln" in name and "weight" in name:
            nn.init.ones_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
        else:
            nn.init.normal_(param, mean=0.0, std=0.02)


# ==============================================================================
# Pure NumPy Implementation (Standalone, zero-dependency reference)
# ==============================================================================
def gelu_np(x):
    """GELU tanh approximation (exact match to PyTorch approximate='tanh')"""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3))))


def layernorm_np(x, weight, bias, eps=LAYER_NORM_EPS):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    norm = (x - mean) / np.sqrt(var + eps)
    return norm * weight + bias


def softmax_np(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class NumpyTransformer:
    """
    Pure NumPy implementation for reference verification without requiring PyTorch.
    Weights are loaded from a dictionary of NumPy arrays.
    """
    def __init__(self, weights):
        self.weights = weights

    def partition_a(self, token_ids):
        """Executes wte, wpe, and layers 0, 1, 2"""
        T = len(token_ids)
        # Embedding
        tok_emb = self.weights["wte.weight"][token_ids]  # [T, D]
        pos_emb = self.weights["wpe.weight"][:T]        # [T, D]
        x = tok_emb + pos_emb

        for l in range(3):
            x = self._layer_forward(x, l)
        return x

    def partition_b(self, activations):
        """Executes layers 3, 4, 5, ln_f, and lm_head"""
        x = activations
        for l in range(3, 6):
            x = self._layer_forward(x, l)
        # ln_f
        x = layernorm_np(x, self.weights["ln_f.weight"], self.weights["ln_f.bias"])
        # lm_head: [T, D] @ [D, V] -> [T, V]
        logits = x @ self.weights["lm_head.weight"].T
        return logits

    def forward(self, token_ids):
        act = self.partition_a(token_ids)
        logits = self.partition_b(act)
        return logits

    def _layer_forward(self, x, l):
        T, D = x.shape
        # ln_1
        norm1 = layernorm_np(x, self.weights[f"h.{l}.ln_1.weight"], self.weights[f"h.{l}.ln_1.bias"])
        # attn: qkv = norm1 @ c_attn.weight.T + c_attn.bias
        qkv = norm1 @ self.weights[f"h.{l}.attn.c_attn.weight"].T + self.weights[f"h.{l}.attn.c_attn.bias"]
        q, k, v = np.split(qkv, 3, axis=-1)  # each [T, D]

        # Multi-head reshape: [T, 6, 64] -> [6, T, 64]
        q = q.reshape(T, NUM_HEADS, D_HEAD).transpose(1, 0, 2)
        k = k.reshape(T, NUM_HEADS, D_HEAD).transpose(1, 0, 2)
        v = v.reshape(T, NUM_HEADS, D_HEAD).transpose(1, 0, 2)

        # Scaled dot product
        scale = 1.0 / math.sqrt(D_HEAD)
        scores = np.matmul(q, k.transpose(0, 2, 1)) * scale  # [6, T, T]

        # Causal mask: upper triangle set to -inf
        mask = np.triu(np.full((T, T), -np.inf), k=1)
        scores = scores + mask[np.newaxis, :, :]

        attn_weights = softmax_np(scores, axis=-1)
        attn_out = np.matmul(attn_weights, v)  # [6, T, 64]
        attn_out = attn_out.transpose(1, 0, 2).reshape(T, D)

        # c_proj
        attn_proj = attn_out @ self.weights[f"h.{l}.attn.c_proj.weight"].T + self.weights[f"h.{l}.attn.c_proj.bias"]
        x = x + attn_proj

        # ln_2
        norm2 = layernorm_np(x, self.weights[f"h.{l}.ln_2.weight"], self.weights[f"h.{l}.ln_2.bias"])
        # mlp
        fc = norm2 @ self.weights[f"h.{l}.mlp.c_fc.weight"].T + self.weights[f"h.{l}.mlp.c_fc.bias"]
        act = gelu_np(fc)
        proj = act @ self.weights[f"h.{l}.mlp.c_proj.weight"].T + self.weights[f"h.{l}.mlp.c_proj.bias"]
        x = x + proj
        return x
