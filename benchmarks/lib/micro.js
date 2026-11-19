/**
 * Stage 7 — communication-path micro-benchmarks (Node, single process, no network).
 *
 * Each operation is timed ALONE on the same activation bytes, so we can answer "which step of moving an activation costs what".
 * A plain memory copy (Uint8Array.set) is included as a REFERENCE: it is the lower bound for any step that has to touch every byte.
 * Anything far above the memcpy line (for example the checksum) is doing real per-byte work.
 */
"use strict";
const P = require("../../protocol/mesh_protocol");
const { summarize } = require("./stats");
const now = () => performance.now();
const D = 384;

function makeActivation(T) { const a = new Float32Array(T * D); let s = 7; for (let i = 0; i < a.length; i++) { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; a[i] = (s / 4294967296 - 0.5) * 4; } return a; }

function timeIt(fn, { warmup, iterations }) {
  for (let i = 0; i < warmup; i++) fn();
  const xs = []; for (let i = 0; i < iterations; i++) { const t = now(); fn(); xs.push(now() - t); }
  return summarize(xs);
}

function runMicro({ sizes = [1, 8, 35, 128, 512], chunkLimits = [65536, 262144], warmup = 20, iterations = 60 } = {}) {
  const out = [];
  for (const T of sizes) {
    const act = makeActivation(T);
    const tensorBytes = act.byteLength;
    const pkt = P.encodeActivation({ sessionId: 1n, seq: 1, activations: act, shape: [1, T, D] });
    const u8 = new Uint8Array(pkt);
    const scratch = new Uint8Array(pkt.byteLength);
    const it = T >= 512 ? Math.max(15, iterations >> 2) : iterations;       // fewer repeats for the long ones to keep the run short
    const opts = { warmup, iterations: it };
    const rec = { T, tensorBytes, packetBytes: pkt.byteLength, headerBytes: P.HEADER_SIZE, ops: {}, chunking: [] };
    rec.ops.memcpyReferenceMs = timeIt(() => scratch.set(u8), opts);
    rec.ops.encodeActivationMs = timeIt(() => P.encodeActivation({ sessionId: 1n, seq: 1, activations: act, shape: [1, T, D] }), opts);
    rec.ops.adler32Ms = timeIt(() => P.adler32(u8, P.HEADER_SIZE, tensorBytes), opts);
    rec.ops.decodePacketMs = timeIt(() => P.decodePacket(pkt), opts);
    const dec = P.decodePacket(pkt);
    rec.ops.validateActivationMs = timeIt(() => P.validateActivation(dec, { dModel: D, maxSeqLen: 512 }), opts);
    rec.ops.partitionBInputCopyMs = timeIt(() => new Float32Array(dec.payload), opts);   // the defensive copy MeshPartitionB.forward makes
    for (const limit of chunkLimits) {
      const frames = P.splitIntoChunks(pkt, limit, 1);
      const c = { maxMessageSize: limit, frames: frames.length, overheadBytes: frames.length > 1 ? frames.length * P.CHUNK_HEADER_SIZE : 0 };
      c.splitMs = timeIt(() => P.splitIntoChunks(pkt, limit, 1), opts);
      c.reassembleMs = frames.length > 1 ? timeIt(() => { const r = new P.ChunkReassembler(); let whole = null; for (const f of frames) whole = r.push(f) || whole; if (!whole) throw new Error("no reassembly"); }, opts) : null;
      rec.chunking.push(c);
    }
    // derived throughput in MB/s (1 MB = 1e6 bytes) from MEDIAN times
    const mbps = (bytes, ms) => (bytes / 1e6) / (ms / 1000);
    rec.throughputMBps = {
      memcpy: mbps(pkt.byteLength, rec.ops.memcpyReferenceMs.median), encode: mbps(pkt.byteLength, rec.ops.encodeActivationMs.median),
      adler32: mbps(tensorBytes, rec.ops.adler32Ms.median), decode: mbps(pkt.byteLength, rec.ops.decodePacketMs.median),
    };
    out.push(rec);
  }
  return out;
}
module.exports = { runMicro, makeActivation };
