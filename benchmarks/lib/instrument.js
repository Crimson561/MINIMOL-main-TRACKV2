/**
 * Stage 7 — non-invasive timing of the distributed pipeline.
 *
 * We do NOT edit engine/mesh_pipeline.js. mesh_pipeline.js calls P.encodeActivation(...), P.splitIntoChunks(...), P.decodePacket(...) etc.
 * by looking them up on the protocol module object at call time, so we can temporarily replace those properties with timing wrappers
 * (and wrap engine.forward and transport.send on the instances we own). Original functions are restored by uninstall().
 *
 * Every wrapped span is a SYNCHRONOUS call that does not contain another wrapped span, so spans are disjoint and their sum can never
 * exceed the wall-clock total. Whatever is not inside a span (event-loop hops, promise scheduling, Set/Map bookkeeping, JSON in the
 * pipeline itself) shows up as "unattributed" = total - sum(spans). That number is reported, never hidden.
 *
 * Cost of measuring: each wrapper adds two performance.now() calls (well under 1 microsecond each in V8). Compare that with the
 * 0.1 .. 5 ms spans we measure for protocol work: the probe effect is small but not zero, and is stated in BENCHMARKS.md.
 */
"use strict";
const P = require("../../protocol/mesh_protocol");
const now = () => performance.now();

const PHASES = [
  "aCompute", "aSerialize", "aChunk", "aSend",
  "bReassemble", "bDecode", "bValidate", "bCompute", "bSerializeReply", "bChunkReply", "bSend",
  "aReassembleReply", "aDecodeReply",
];

class Instrument {
  constructor() { this.saved = []; this.acc = null; this.counts = null; this.installed = false; }

  _patch(obj, key, wrapperFactory) { const orig = obj[key]; this.saved.push([obj, key, orig]); obj[key] = wrapperFactory(orig); }

  begin() { this.acc = Object.fromEntries(PHASES.map((p) => [p, 0])); this.counts = { framesAB: 0, framesBA: 0, bytesAB: 0, bytesBA: 0 }; }
  end() { const r = { spans: this.acc, counts: this.counts }; this.acc = null; return r; }
  _add(phase, t0) { if (this.acc) this.acc[phase] += now() - t0; }

  /** @param {{engineA, engineB, pipeA, pipeB}} h  objects whose methods get wrapped (instances, not classes) */
  install({ engineA, engineB, transportA, transportB }) {
    if (this.installed) throw new Error("already installed");
    const self = this;

    this._patch(P, "encodeActivation", (orig) => function (...a) { const t = now(); try { return orig.apply(this, a); } finally { self._add("aSerialize", t); } });
    this._patch(P, "encodeToken", (orig) => function (...a) { const t = now(); try { return orig.apply(this, a); } finally { self._add("bSerializeReply", t); } });
    this._patch(P, "validateActivation", (orig) => function (...a) { const t = now(); try { return orig.apply(this, a); } finally { self._add("bValidate", t); } });
    this._patch(P, "decodePacket", (orig) => function (...a) {
      const t = now(); let r;
      try { r = orig.apply(this, a); return r; }
      finally { const dt = now() - t; if (self.acc) { const k = r && r.msgType === P.MSG_TYPES.ACTIVATION ? "bDecode" : r && r.msgType === P.MSG_TYPES.TOKEN ? "aDecodeReply" : null; if (k) self.acc[k] += dt; } }
    });
    // splitIntoChunks: the activation request goes first, then the reply. Classify by the message type byte (offset 5) of the packet.
    this._patch(P, "splitIntoChunks", (orig) => function (packet, ...rest) {
      const t = now(); try { return orig.call(this, packet, ...rest); }
      finally { const type = new Uint8Array(packet)[5]; self._add(type === P.MSG_TYPES.ACTIVATION ? "aChunk" : type === P.MSG_TYPES.TOKEN ? "bChunkReply" : "aChunk", t); }
    });
    // Reassembly: A->B messages use B's reassembler, B->A replies use A's. Distinguish by whose reassembler instance it is (tagged by the harness).
    this._patch(P.ChunkReassembler.prototype, "push", (orig) => function (raw) {
      const t = now(); try { return orig.call(this, raw); } finally { self._add(this.__owner === "A" ? "aReassembleReply" : "bReassemble", t); }
    });
    // engines
    const wrapEngine = (eng, phase) => { const orig = eng.forward; this.saved.push([eng, "forward", orig]); eng.forward = function (...a) { const t = now(); try { return orig.apply(this, a); } finally { self._add(phase, t); } }; };
    wrapEngine(engineA, "aCompute"); wrapEngine(engineB, "bCompute");
    // transports: time of the send() call itself (loopback: a memcpy; DataChannel: queueing) + count frames/bytes
    const wrapTransport = (tr, phase, fk, bk) => { const orig = tr.send; this.saved.push([tr, "send", orig]); tr.send = function (buf) { const t = now(); if (self.counts) { self.counts[fk]++; self.counts[bk] += buf.byteLength; } try { return orig.call(this, buf); } finally { self._add(phase, t); } }; };
    wrapTransport(transportA, "aSend", "framesAB", "bytesAB"); wrapTransport(transportB, "bSend", "framesBA", "bytesBA");
    this.installed = true;
  }

  uninstall() { for (let i = this.saved.length - 1; i >= 0; i--) { const [o, k, f] = this.saved[i]; o[k] = f; } this.saved = []; this.installed = false; }
}

module.exports = { Instrument, PHASES };
