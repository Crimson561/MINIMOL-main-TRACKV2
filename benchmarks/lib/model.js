/** Stage 7 — load the exported weights exactly like tests/test_stage5_pipeline.js does. */
"use strict";
const fs = require("fs");
const path = require("path");
const ROOT = path.join(__dirname, "..", "..");
const E_DEFAULT = require("../../engine/inference_engine");
const Pl = require("../../engine/mesh_pipeline");
const WEIGHTS_DIR = path.join(ROOT, "room-server", "public", "weights");

/** @param {object} [E] engine module to build the partitions from (default: the current engine; Stage 7 passes the saved baseline engine for before/after runs) */
function loadModel(E = E_DEFAULT) {
  const mp = path.join(WEIGHTS_DIR, "weights_manifest.json");
  if (!fs.existsSync(mp)) throw new Error("Weights missing. Run: python model/export_weights.py");
  const manifest = JSON.parse(fs.readFileSync(mp, "utf-8"));
  const ab = (f) => { const b = fs.readFileSync(path.join(WEIGHTS_DIR, f)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
  const bufA = ab("partition_a.bin"), bufB = ab("partition_b.bin"), bufFull = ab("full_model.bin");
  Pl.verifyPartitionBuffer(bufA, manifest, "a"); Pl.verifyPartitionBuffer(bufB, manifest, "b");   // refuse to benchmark the wrong weights
  return {
    manifest, fingerprint: Pl.modelFingerprint(manifest),
    partitionA: E.MeshPartitionA.fromBuffer(bufA, manifest),
    partitionB: E.MeshPartitionB.fromBuffer(bufB, manifest),
    full: E.MeshFullModel.fromBuffer(bufFull, manifest),
    info: {
      architecture: manifest.architecture, layers: manifest.num_layers, dModel: manifest.d_model, heads: manifest.num_heads, dFF: manifest.d_ff,
      vocab: manifest.vocab_size, contextLength: manifest.context_length, parameters: manifest.total_parameters, seed: manifest.seed,
      partitionAAdler32: manifest.partition_a.checksum_adler32, partitionBAdler32: manifest.partition_b.checksum_adler32,
      partitionABytes: bufA.byteLength, partitionBBytes: bufB.byteLength, fullBytes: bufFull.byteLength, precision: "float32",
      weightsAreTrained: false,
    },
  };
}

/** The fixed, deterministic inputs every benchmark uses, so runs are comparable. */
const PROMPT = [77, 69, 83, 72];               // "MESH"
const GEN_TOKENS = 10;                          // matches the PyTorch reference generation in TESTING.md
const REFERENCE_GENERATION = [77, 69, 83, 72, 228, 228, 228, 228, 228, 228, 98, 98, 240, 98];
const fixedTokens = (T) => Array.from({ length: T }, (_, i) => (i * 37 + 11) & 255);

module.exports = { loadModel, PROMPT, GEN_TOKENS, REFERENCE_GENERATION, fixedTokens, WEIGHTS_DIR, ROOT };
