/**
 * MESH Browser-Compatible CPU Inference Engine
 * Pure JavaScript, zero external dependencies, Float32Array operations.
 * Supports Android Chrome, iOS Safari, Node.js.
 */

"use strict";

const VOCAB_SIZE = 256;
const MAX_SEQ_LEN = 512;
const NUM_LAYERS = 6;
const D_MODEL = 384;
const NUM_HEADS = 6;
const D_HEAD = 64; // 384 / 6
const D_FF = 1536;
const LAYER_NORM_EPS = 1e-5;
const SQRT_2_OVER_PI = Math.sqrt(2.0 / Math.PI);

// =============================================================================
// Math & Tensor Kernels (Optimized CPU operations)
// =============================================================================

/**
 * 2D Matrix Multiplication: C = A @ B_T + bias
 * A: [M, K]
 * B_T: [K, N] (stored as B with shape [N, K], so dot product is with row n of B)
 * bias: [N] (optional)
 * C: [M, N]
 */
function matmulTransposeB(A, B, bias, M, K, N, out) {
  const result = out || new Float32Array(M * N);
  for (let m = 0; m < M; m++) {
    const aOffset = m * K;
    const cOffset = m * N;
    for (let n = 0; n < N; n++) {
      const bOffset = n * K;
      let sum = bias ? bias[n] : 0.0;
      for (let k = 0; k < K; k++) {
        sum += A[aOffset + k] * B[bOffset + k];
      }
      result[cOffset + n] = sum;
    }
  }
  return result;
}

/**
 * Standard 2D Matrix Multiplication: C = A @ B
 * A: [M, K]
 * B: [K, N]
 * C: [M, N]
 */
function matmul(A, B, M, K, N, out) {
  const result = out || new Float32Array(M * N);
  result.fill(0);
  for (let m = 0; m < M; m++) {
    const aOffset = m * K;
    const cOffset = m * N;
    for (let k = 0; k < K; k++) {
      const aVal = A[aOffset + k];
      const bOffset = k * N;
      for (let n = 0; n < N; n++) {
        result[cOffset + n] += aVal * B[bOffset + n];
      }
    }
  }
  return result;
}

/**
 * LayerNorm over last dimension D
 * X: [T, D]
 */
function layerNorm(X, weight, bias, T, D, out) {
  const result = out || new Float32Array(T * D);
  for (let t = 0; t < T; t++) {
    const offset = t * D;
    // Mean
    let sum = 0.0;
    for (let d = 0; d < D; d++) {
      sum += X[offset + d];
    }
    const mean = sum / D;

    // Variance
    let varSum = 0.0;
    for (let d = 0; d < D; d++) {
      const diff = X[offset + d] - mean;
      varSum += diff * diff;
    }
    const invStd = 1.0 / Math.sqrt(varSum / D + LAYER_NORM_EPS);

    // Normalize and scale
    for (let d = 0; d < D; d++) {
      result[offset + d] = (X[offset + d] - mean) * invStd * weight[d] + bias[d];
    }
  }
  return result;
}

/**
 * GELU with tanh approximation (GPT-2 standard)
 * 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
function gelu(X, length, out) {
  const result = out || new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const x = X[i];
    const inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
    result[i] = 0.5 * x * (1.0 + Math.tanh(inner));
  }
  return result;
}

/**
 * In-place Softmax over last dimension
 */
function softmaxInPlace(arr, offset, length) {
  let maxVal = -Infinity;
  for (let i = 0; i < length; i++) {
    if (arr[offset + i] > maxVal) maxVal = arr[offset + i];
  }
  let sumExp = 0.0;
  for (let i = 0; i < length; i++) {
    const expVal = Math.exp(arr[offset + i] - maxVal);
    arr[offset + i] = expVal;
    sumExp += expVal;
  }
  const invSum = 1.0 / sumExp;
  for (let i = 0; i < length; i++) {
    arr[offset + i] *= invSum;
  }
}

/**
 * Multi-Head Causal Self-Attention block
 */
function causalSelfAttention(X, weights, layerIdx, T) {
  const prefix = `h.${layerIdx}.attn`;
  const c_attn_w = weights[`${prefix}.c_attn.weight`];
  const c_attn_b = weights[`${prefix}.c_attn.bias`];
  const c_proj_w = weights[`${prefix}.c_proj.weight`];
  const c_proj_b = weights[`${prefix}.c_proj.bias`];

  // 1. QKV projection: [T, 384] @ [384, 1152] -> [T, 1152]
  const qkv = matmulTransposeB(X, c_attn_w, c_attn_b, T, D_MODEL, 3 * D_MODEL);

  // 2. Multi-head attention across 6 heads
  const attnOut = new Float32Array(T * D_MODEL);
  const scale = 1.0 / Math.sqrt(D_HEAD);
  const scores = new Float32Array(T);

  for (let h = 0; h < NUM_HEADS; h++) {
    const headOffset = h * D_HEAD;

    for (let i = 0; i < T; i++) {
      // Query vector for head h at position i
      const qOffset = i * (3 * D_MODEL) + headOffset;

      // Compute dot products with all keys up to position i (causal mask)
      for (let j = 0; j <= i; j++) {
        const kOffset = j * (3 * D_MODEL) + D_MODEL + headOffset;
        let dot = 0.0;
        for (let d = 0; d < D_HEAD; d++) {
          dot += qkv[qOffset + d] * qkv[kOffset + d];
        }
        scores[j] = dot * scale;
      }

      // Softmax over positions 0..i
      softmaxInPlace(scores, 0, i + 1);

      // Weighted sum of values 0..i
      const outOffset = i * D_MODEL + headOffset;
      for (let d = 0; d < D_HEAD; d++) {
        let valSum = 0.0;
        for (let j = 0; j <= i; j++) {
          const vOffset = j * (3 * D_MODEL) + 2 * D_MODEL + headOffset;
          valSum += scores[j] * qkv[vOffset + d];
        }
        attnOut[outOffset + d] = valSum;
      }
    }
  }

  // 3. Output projection: [T, 384] @ [384, 384] -> [T, 384]
  return matmulTransposeB(attnOut, c_proj_w, c_proj_b, T, D_MODEL, D_MODEL);
}

/**
 * MLP / Feed-Forward Block: LayerNorm -> FC (1536) -> GELU -> Proj (384)
 */
function mlpBlock(X, weights, layerIdx, T) {
  const prefix = `h.${layerIdx}.mlp`;
  const c_fc_w = weights[`${prefix}.c_fc.weight`];
  const c_fc_b = weights[`${prefix}.c_fc.bias`];
  const c_proj_w = weights[`${prefix}.c_proj.weight`];
  const c_proj_b = weights[`${prefix}.c_proj.bias`];

  // 1. FC: [T, 384] @ [384, 1536] -> [T, 1536]
  const fc = matmulTransposeB(X, c_fc_w, c_fc_b, T, D_MODEL, D_FF);

  // 2. GELU
  const act = gelu(fc, T * D_FF);

  // 3. Proj: [T, 1536] @ [1536, 384] -> [T, 384]
  return matmulTransposeB(act, c_proj_w, c_proj_b, T, D_FF, D_MODEL);
}

/**
 * Forward single Transformer layer
 */
function forwardLayer(X, weights, layerIdx, T) {
  const ln_1_w = weights[`h.${layerIdx}.ln_1.weight`];
  const ln_1_b = weights[`h.${layerIdx}.ln_1.bias`];
  const ln_2_w = weights[`h.${layerIdx}.ln_2.weight`];
  const ln_2_b = weights[`h.${layerIdx}.ln_2.bias`];

  // 1. Attention with residual
  const norm1 = layerNorm(X, ln_1_w, ln_1_b, T, D_MODEL);
  const attn = causalSelfAttention(norm1, weights, layerIdx, T);
  for (let i = 0; i < T * D_MODEL; i++) {
    X[i] += attn[i];
  }

  // 2. MLP with residual
  const norm2 = layerNorm(X, ln_2_w, ln_2_b, T, D_MODEL);
  const mlp = mlpBlock(norm2, weights, layerIdx, T);
  for (let i = 0; i < T * D_MODEL; i++) {
    X[i] += mlp[i];
  }

  return X;
}

// =============================================================================
// Helper: Map Raw Binary Buffer to Named Tensors
// =============================================================================
function mapWeightsFromBuffer(buffer, tensorMeta) {
  const weights = {};
  for (const [name, meta] of Object.entries(tensorMeta)) {
    weights[name] = new Float32Array(buffer, meta.byte_offset, meta.num_elements);
  }
  return weights;
}

// =============================================================================
// Phone A Engine (Partition A: Embeddings + Layers 0, 1, 2)
// =============================================================================
class MeshPartitionA {
  constructor(weights) {
    this.weights = weights; // named Float32Array dict
  }

  static fromBuffer(buffer, manifest) {
    const meta = manifest.partition_a ? manifest.partition_a.tensors : manifest.tensors;
    return new MeshPartitionA(mapWeightsFromBuffer(buffer, meta));
  }

  /**
   * Forward pass on Phone A:
   * @param {Array<number>|Uint8Array} tokenIds - prompt byte sequence
   * @returns {Float32Array} Intermediate activation tensor [1, T, 384]
   */
  forward(tokenIds) {
    const T = tokenIds.length;
    if (T > MAX_SEQ_LEN) {
      throw new Error(`Token sequence length ${T} exceeds max context length ${MAX_SEQ_LEN}`);
    }

    const wte = this.weights["wte.weight"];
    const wpe = this.weights["wpe.weight"];

    // 1. Embedding lookup: tok_emb + pos_emb -> [T, 384]
    const X = new Float32Array(T * D_MODEL);
    for (let t = 0; t < T; t++) {
      const tok = tokenIds[t];
      const tokOffset = tok * D_MODEL;
      const posOffset = t * D_MODEL;
      const xOffset = t * D_MODEL;
      for (let d = 0; d < D_MODEL; d++) {
        X[xOffset + d] = wte[tokOffset + d] + wpe[posOffset + d];
      }
    }

    // 2. Transformer layers 0, 1, 2
    for (let l = 0; l < 3; l++) {
      forwardLayer(X, this.weights, l, T);
    }

    return X; // Activations
  }
}

// =============================================================================
// Phone B Engine (Partition B: Layers 3, 4, 5 + ln_f + lm_head)
// =============================================================================
class MeshPartitionB {
  constructor(weights) {
    this.weights = weights;
  }

  static fromBuffer(buffer, manifest) {
    const meta = manifest.partition_b ? manifest.partition_b.tensors : manifest.tensors;
    return new MeshPartitionB(mapWeightsFromBuffer(buffer, meta));
  }

  /**
   * Forward pass on Phone B:
   * @param {Float32Array} activations - intermediate activations from Phone A [1, T, 384]
   * @param {number} T - sequence length
   * @returns {{ nextByte: number, logits: Float32Array }}
   */
  forward(activations, T) {
    // Copy activations to prevent mutating input buffer
    const X = new Float32Array(activations);

    // 1. Transformer layers 3, 4, 5
    for (let l = 3; l < 6; l++) {
      forwardLayer(X, this.weights, l, T);
    }

    // 2. Final LayerNorm: ln_f
    const norm = layerNorm(X, this.weights["ln_f.weight"], this.weights["ln_f.bias"], T, D_MODEL);

    // 3. Output Projection (lm_head) for the LAST token position only
    const lastTokOffset = (T - 1) * D_MODEL;
    const lastTokenVector = norm.subarray(lastTokOffset, lastTokOffset + D_MODEL);

    const lm_head_w = this.weights["lm_head.weight"]; // [256, 384]
    const logits = new Float32Array(VOCAB_SIZE);

    let maxVal = -Infinity;
    let nextByte = 0;

    for (let v = 0; v < VOCAB_SIZE; v++) {
      const wOffset = v * D_MODEL;
      let sum = 0.0;
      for (let d = 0; d < D_MODEL; d++) {
        sum += lastTokenVector[d] * lm_head_w[wOffset + d];
      }
      logits[v] = sum;
      if (sum > maxVal) {
        maxVal = sum;
        nextByte = v;
      }
    }

    return { nextByte, logits };
  }
}

// =============================================================================
// Full Model Reference Engine (Single Node)
// =============================================================================
class MeshFullModel {
  constructor(weights) {
    this.partitionA = new MeshPartitionA(weights);
    this.partitionB = new MeshPartitionB(weights);
  }

  static fromBuffer(buffer, manifest) {
    const meta = manifest.tensors || manifest.full_model.tensors;
    return new MeshFullModel(mapWeightsFromBuffer(buffer, meta));
  }

  forward(tokenIds) {
    const act = this.partitionA.forward(tokenIds);
    return this.partitionB.forward(act, tokenIds.length);
  }

  generate(promptBytes, maxNewTokens = 10) {
    const tokens = Array.from(promptBytes);
    for (let i = 0; i < maxNewTokens; i++) {
      const { nextByte } = this.forward(tokens);
      tokens.push(nextByte);
    }
    return tokens;
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    VOCAB_SIZE,
    MAX_SEQ_LEN,
    NUM_LAYERS,
    D_MODEL,
    NUM_HEADS,
    D_HEAD,
    D_FF,
    MeshPartitionA,
    MeshPartitionB,
    MeshFullModel,
    layerNorm,
    matmulTransposeB,
    gelu,
    softmaxInPlace,
  };
} else if (typeof window !== "undefined") {
  window.MeshInference = {
    VOCAB_SIZE,
    MAX_SEQ_LEN,
    NUM_LAYERS,
    D_MODEL,
    NUM_HEADS,
    D_HEAD,
    D_FF,
    MeshPartitionA,
    MeshPartitionB,
    MeshFullModel,
  };
}
