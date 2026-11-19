/**
 * MINIMOL Stage 5: activation-transfer pipeline (coordination between Device A and Device B).
 *
 * Transport-agnostic: give it { send(ArrayBuffer) -> Promise|void, maxMessageSize() -> number } and call
 * handleMessage(arrayBuffer) for every message the DataChannel delivers. The same file runs in Node tests and browsers.
 *
 * One inference request = one session:
 *   A: tokens -> partition A (embeddings + layers 0..2) -> activation [1,T,384]
 *      -> ACTIVATION packet {sessionId, seq=1} (chunked if larger than the DataChannel limit)
 *   B: reassemble -> decodePacket (checksum) -> validateActivation -> once-only check -> partition B (layers 3..5 + head)
 *      -> TOKEN packet {same sessionId, same seq} (+ logits when A asked for them)   or   ERROR packet {code, message}
 *   A: accepts a reply only if sessionId and seq match the request in flight.
 */
"use strict";

(function (root) {
  const isNode = typeof module !== "undefined" && module.exports;
  const P = isNode ? require("../protocol/mesh_protocol") : root.MeshProtocol;
  const E = isNode ? require("./inference_engine") : root.MeshInference;
  const { MSG_TYPES, MeshError, HEADER_SIZE } = P;
  const { D_MODEL, MAX_SEQ_LEN, VOCAB_SIZE } = E;

  const SPLIT_AFTER_LAYER = 2; // the engine hard-codes A = layers 0..2, B = layers 3..5; both sides must agree on the cut
  const SEEN_LIMIT = 4096;
  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

  /* ---------- weight / model consistency ---------- */

  /** Adler-32 of a partition file's bytes: the same algorithm export_weights.py records in the manifest. */
  function partitionChecksum(buffer) {
    return P.adler32(new Uint8Array(buffer), 0, buffer.byteLength);
  }

  /** Refuse a downloaded partition that is not the exact file the manifest describes. */
  function verifyPartitionBuffer(buffer, manifest, which) {
    const meta = manifest && manifest["partition_" + which];
    if (!meta || meta.checksum_adler32 === undefined) throw new MeshError("MODEL_MISMATCH", "weights_manifest.json has no partition checksums: re-run model/export_weights.py");
    if (buffer.byteLength !== meta.total_bytes) throw new MeshError("MODEL_MISMATCH", `partition ${which} is ${buffer.byteLength} bytes, manifest says ${meta.total_bytes}`);
    const got = partitionChecksum(buffer);
    if (got !== meta.checksum_adler32) throw new MeshError("MODEL_MISMATCH", `partition ${which} checksum ${got} does not match manifest ${meta.checksum_adler32}`);
    return got;
  }

  /** One string describing the whole model (architecture + both partitions' checksums + the cut). Both devices must send equal strings. */
  function modelFingerprint(m) {
    const a = m.partition_a && m.partition_a.checksum_adler32, b = m.partition_b && m.partition_b.checksum_adler32;
    if (a === undefined || b === undefined) throw new MeshError("MODEL_MISMATCH", "weights_manifest.json has no partition checksums: re-run model/export_weights.py");
    return `${m.architecture}|vocab${m.vocab_size}|ctx${m.context_length}|L${m.num_layers}|d${m.d_model}|h${m.num_heads}|ff${m.d_ff}|seed${m.seed}|params${m.total_parameters}|cut${SPLIT_AFTER_LAYER}|a${a}|b${b}`;
  }

  /* ---------- shared base: HELLO handshake, chunked send, receive/reassemble/decode, dispatch ---------- */

  class PipelineBase {
    constructor({ role, engine = null, fingerprint, transport, timeoutMs = 20000, reassemblyTimeoutMs = 10000, onEvent = null }) {
      this.role = role; this.engine = engine; this.fingerprint = fingerprint; this.transport = transport;
      this.timeoutMs = timeoutMs; this.onEvent = onEvent;
      this.reassembler = new P.ChunkReassembler();
      this.messageId = 0;
      this.peer = null;          // result of validating the other device's HELLO: { ok, fatal, reason }
      this.waiters = [];
      this.stats = { framesSent: 0, received: 0, expired: 0 };
      // Stage 6: lifecycle. A disposed pipeline is dead for good: it owns no timers, rejects new work and ignores incoming messages.
      this.disposed = false; this.disposedReason = null;
      this.reassemblyTimeoutMs = reassemblyTimeoutMs; this.reassemblyTimer = null;
    }

    /** Stage 6: stop everything this pipeline owns. Safe to call any number of times; only the first call does anything. */
    dispose(reason = "pipeline disposed") {
      if (this.disposed) return false;
      this.disposed = true; this.disposedReason = reason;
      this._clearReassemblyTimer(); this.reassembler.reset();
      const waiters = this.waiters; this.waiters = [];
      for (const w of waiters) w.reject(new MeshError("CHANNEL_CLOSED", reason));   // each waiter clears its own timer
      this._onDispose(new MeshError("CHANNEL_CLOSED", reason));
      this._emit("disposed", { reason });
      this.onEvent = null;                                                          // an old pipeline must never write into a newer session's log/UI
      return true;
    }
    _onDispose() {}
    _assertAlive() { if (this.disposed) throw new MeshError("CHANNEL_CLOSED", this.disposedReason || "pipeline disposed"); }

    /** Stage 6: a chunked message that stops arriving is discarded after reassemblyTimeoutMs of silence (inactivity, re-armed by each chunk). */
    _armReassemblyTimer() {
      this._clearReassemblyTimer();
      if (!(this.reassemblyTimeoutMs > 0) || !this.reassembler.inProgress) return;
      const t = setTimeout(() => {
        this.reassemblyTimer = null;
        if (this.disposed || !this.reassembler.inProgress) return;
        this.reassembler.reset(); this.stats.expired++;
        this._emit("warn", { message: `discarded an incomplete chunked message: no new chunk for ${this.reassemblyTimeoutMs} ms` });
      }, this.reassemblyTimeoutMs);
      if (t && typeof t.unref === "function") t.unref();     // Node only: a pending expiry must not keep a finished test process alive
      this.reassemblyTimer = t;
    }
    _clearReassemblyTimer() { if (this.reassemblyTimer !== null) { clearTimeout(this.reassemblyTimer); this.reassemblyTimer = null; } }

    _emit(type, info) { if (this.onEvent) { try { this.onEvent(type, info || {}); } catch (e) { /* a logging callback must never break the pipeline */ } } }

    /** Send one finished MESH packet, cut into chunks only if it exceeds the DataChannel limit. */
    async sendPacket(packet) {
      this._assertAlive();
      const frames = P.splitIntoChunks(packet, this.transport.maxMessageSize(), ++this.messageId);
      for (const f of frames) { this._assertAlive(); await this.transport.send(f); }   // re-checked per frame: dispose() may happen while a big transfer is under way
      this.stats.framesSent += frames.length;
      return frames.length;
    }

    sendHello() {
      return this.sendPacket(P.encodeControl({ sessionId: 0, command: "HELLO", data: { role: this.role, loaded: !!this.engine, fingerprint: this.fingerprint } }));
    }

    _onHello(p) {
      const h = p.payload || {};
      let peer;
      if (h.role === this.role) peer = { ok: false, fatal: true, reason: `both devices claim to be Device ${this.role}` };
      else if (!h.loaded) peer = { ok: false, fatal: false, reason: "the other device has not loaded its model partition yet" };
      else if (h.fingerprint !== this.fingerprint) peer = { ok: false, fatal: true, reason: "the two devices do not have the same model/weights (fingerprint mismatch)" };
      else peer = { ok: true };
      this.peer = peer;
      this._emit("hello", peer);
      this.waiters = this.waiters.filter((w) => {
        if (peer.ok) { w.resolve(); return false; }
        if (peer.fatal) { w.reject(new MeshError("MODEL_MISMATCH", peer.reason)); return false; }
        return true;
      });
    }

    /** Resolves once the other device's HELLO has been validated; rejects on mismatch or after ms. */
    waitReady(ms = this.timeoutMs) {
      if (this.peer && this.peer.ok) return Promise.resolve();
      if (this.peer && this.peer.fatal) return Promise.reject(new MeshError("MODEL_MISMATCH", this.peer.reason));
      return new Promise((resolve, reject) => {
        const w = { resolve: () => { clearTimeout(t); resolve(); }, reject: (e) => { clearTimeout(t); reject(e); } };
        const t = setTimeout(() => {
          this.waiters = this.waiters.filter((x) => x !== w);
          reject(new MeshError("NOT_READY", (this.peer && this.peer.reason) || "no HELLO received from the other device"));
        }, ms);
        this.waiters.push(w);
      });
    }

    /** Feed every DataChannel message here. Never throws; problems go to _problem(). */
    async handleMessage(raw) {
      if (this.disposed) return;                              // Stage 6: a dead pipeline ignores whatever is still in flight
      let buf = raw, p;
      try {
        if (ArrayBuffer.isView(raw)) buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
        if (!(buf instanceof ArrayBuffer)) throw new MeshError("BAD_PACKET", "message is not binary");
        if (P.isChunk(buf)) {
          const before = this.reassembler.discarded;
          buf = this.reassembler.push(buf);
          if (this.reassembler.discarded > before) this._emit("warn", { message: "discarded an unfinished chunked message (a new one started)" });
          if (!buf) { this._armReassemblyTimer(); return; }   // still waiting for more chunks: (re)start the expiry clock
          this._clearReassemblyTimer();
        }
        else if (this.reassembler.inProgress) { this._clearReassemblyTimer(); this.reassembler.reset(); this._emit("warn", { message: "discarded an unfinished chunked message (a whole packet arrived before it completed)" }); }
        p = P.decodePacket(buf);
        if (buf.byteLength !== HEADER_SIZE + p.payloadLength) throw new MeshError("BAD_PACKET", `packet is ${buf.byteLength} bytes but header declares ${HEADER_SIZE + p.payloadLength}`);
      } catch (e) {
        this._clearReassemblyTimer();                           // a failed reassembler holds no partial message any more
        const err = e instanceof MeshError ? e : new MeshError(/checksum/i.test(e.message) ? "CHECKSUM" : "BAD_PACKET", e.message);
        return this._problem(err, "0", 0);
      }
      this.stats.received++;
      try {
        switch (p.msgType) {
          case MSG_TYPES.CONTROL:
            if (p.payload && p.payload.command === "HELLO") this._onHello(p);
            else this._emit("warn", { message: "ignored unknown control command " + (p.payload && p.payload.command) });
            break;
          case MSG_TYPES.ACTIVATION: await this._onActivation(p, buf); break;
          case MSG_TYPES.TOKEN: this._onResult(p); break;
          case MSG_TYPES.ERROR: this._onPeerError(p); break;
          default: throw new MeshError("BAD_PACKET", `unexpected message type 0x${p.msgType.toString(16)}`);
        }
      } catch (e) {
        await this._problem(e instanceof MeshError ? e : new MeshError("INTERNAL", e.message), p.sessionId, p.seq);
      }
    }

    _onActivation() { throw new MeshError("UNEXPECTED", `Device ${this.role} does not accept activations`); }
    _onResult() { throw new MeshError("UNEXPECTED", `Device ${this.role} does not accept results`); }
    _onPeerError(p) { this._emit("peer-error", { code: p.payload && p.payload.code, message: p.payload && p.payload.message }); }
    async _problem(err) { this._emit("error", { code: err.code, message: err.message }); }
  }

  /* ---------- Device A: partition A + request owner ---------- */

  class PipelineA extends PipelineBase {
    constructor(opts) { super({ ...opts, role: "A" }); this.pending = null; this.counter = 0; }

    _newSessionId() { return (BigInt(Date.now()) << 16n) | BigInt(++this.counter & 0xffff); }

    /**
     * Run one distributed forward pass: A computes layers 0..2, B computes 3..5 + head, B returns the next byte.
     * @param {ArrayLike<number>} tokenIds bytes 0..255, length 1..512
     * @param {{wantLogits?: boolean, timeoutMs?: number, sessionId?: bigint|number}} opts
     */
    async infer(tokenIds, { wantLogits = false, timeoutMs = this.timeoutMs, sessionId = null } = {}) {
      this._assertAlive();
      if (!this.engine) throw new MeshError("NOT_READY", "partition A is not loaded");
      const T = tokenIds && tokenIds.length;
      if (!Number.isInteger(T) || T < 1 || T > MAX_SEQ_LEN) throw new MeshError("BAD_INPUT", `prompt must be 1..${MAX_SEQ_LEN} tokens, got ${T}`);
      for (let i = 0; i < T; i++) if (!Number.isInteger(tokenIds[i]) || tokenIds[i] < 0 || tokenIds[i] > 255) throw new MeshError("BAD_INPUT", `token ${i} is not a byte 0..255`);
      if (this.pending) throw new MeshError("BUSY", "another inference request is still in flight");

      const sid = sessionId !== null ? BigInt(sessionId) : this._newSessionId();
      const req = { sessionId: String(sid), seq: 1, wantLogits, t0: now(), timer: null, resolve: null, reject: null };
      const done = new Promise((res, rej) => { req.resolve = res; req.reject = rej; });
      done.catch(() => {});                       // settled errors are delivered through the awaited promise below
      this.pending = req;
      req.timer = setTimeout(() => {
        const ready = this.peer && this.peer.ok;      // never got a valid HELLO: say why, instead of a bare timeout
        this._settle(req, ready ? new MeshError("TIMEOUT", `no reply from Device B within ${timeoutMs} ms (session ${req.sessionId})`)
          : new MeshError("NOT_READY", (this.peer && this.peer.reason) || "no HELLO received from Device B"), true);
      }, timeoutMs);
      try {
        await this.waitReady(timeoutMs);          // B has loaded the same model and said so
        let t = now();
        const activation = this.engine.forward(tokenIds);   // <- the real intermediate activation, [T*384] floats after layer 2
        req.computeMsA = now() - t;
        const packet = P.encodeActivation({ sessionId: sid, seq: req.seq, activations: activation, shape: [1, T, D_MODEL], flags: wantLogits ? P.FLAG_WANT_LOGITS : 0 });
        req.T = T; req.activationBytes = packet.byteLength; req.tSend = now();
        req.frames = await this.sendPacket(packet);
        this._emit("sent", { sessionId: req.sessionId, T, bytes: packet.byteLength, frames: req.frames });
        return await done;
      } catch (e) {
        this._settle(req, e, true);
        return done;
      }
    }

    _onDispose(err) { if (this.pending) this._settle(this.pending, err, true); }   // Stage 6: the request in flight fails NOW, not at its timeout

    _settle(req, value, isError) {
      if (this.pending !== req) return;
      clearTimeout(req.timer); this.pending = null;
      if (isError) req.reject(value); else req.resolve(value);
    }

    _onResult(p) {
      const req = this.pending;
      if (!req || p.sessionId !== req.sessionId || p.seq !== req.seq) {
        this._emit("warn", { message: `ignored a reply for session ${p.sessionId} seq ${p.seq}: no matching request in flight` });
        return;
      }
      const r = p.payload;
      if (!r || !Number.isInteger(r.token) || r.token < 0 || r.token > 255) throw new MeshError("BAD_REPLY", "reply carries no valid next byte");
      let logits = null;
      if (req.wantLogits) {
        if (!Array.isArray(r.logits) || r.logits.length !== VOCAB_SIZE) throw new MeshError("BAD_REPLY", `reply should carry ${VOCAB_SIZE} logits`);
        logits = Float32Array.from(r.logits);
        let best = 0;
        for (let i = 0; i < VOCAB_SIZE; i++) {
          if (!Number.isFinite(logits[i])) throw new MeshError("BAD_REPLY", `logit ${i} is not finite`);
          if (logits[i] > logits[best]) best = i;
        }
        if (best !== r.token) throw new MeshError("BAD_REPLY", `reply token ${r.token} is not the argmax of its logits (${best})`);
      }
      this._settle(req, {
        sessionId: req.sessionId, seq: req.seq, T: req.T, token: r.token, logits,
        computeMsA: req.computeMsA, computeMsB: r.timing && r.timing.computeMs,
        rttMs: now() - req.tSend, totalMs: now() - req.t0, activationBytes: req.activationBytes, frames: req.frames,
      }, false);
    }

    _onPeerError(p) {
      super._onPeerError(p);
      const req = this.pending, e = p.payload || {};
      if (req && (p.sessionId === req.sessionId || p.sessionId === "0")) {
        const err = new MeshError(e.code || "PEER_ERROR", `Device B: ${e.message || "unknown error"}`);
        err.fromPeer = true;
        this._settle(req, err, true);
      }
    }

    async _problem(err, sessionId) {
      this._emit("error", { code: err.code, message: err.message });
      const req = this.pending;
      if (req && (sessionId === req.sessionId || sessionId === "0")) this._settle(req, err, true);   // never wait out a timeout for a known failure
    }
  }

  /* ---------- Device B: partition B, processes each activation exactly once ---------- */

  class PipelineB extends PipelineBase {
    constructor(opts) { super({ ...opts, role: "B" }); this.seen = new Set(); this.stats.processed = 0; this.stats.rejected = 0; }

    async _onActivation(p, buf) {
      if (!this.engine) throw new MeshError("NOT_READY", "partition B is not loaded");
      if (!this.peer || !this.peer.ok) throw new MeshError("NOT_READY", "model/weights of Device A not confirmed by HELLO");
      const { T } = P.validateActivation(p, { dModel: D_MODEL, maxSeqLen: MAX_SEQ_LEN });   // before any compute
      const key = `${p.sessionId}:${p.seq}`;
      if (this.seen.has(key)) throw new MeshError("DUPLICATE", `activation for session ${p.sessionId} seq ${p.seq} was already processed`);
      this.seen.add(key);
      if (this.seen.size > SEEN_LIMIT) this.seen.delete(this.seen.values().next().value);

      this._emit("activation", { sessionId: p.sessionId, seq: p.seq, T, bytes: buf.byteLength });
      const t = now();
      let r;
      try { r = this.engine.forward(p.payload, T); }      // layers 3..5 + ln_f + lm_head; the engine copies the input, it never writes into the received buffer
      catch (e) { throw new MeshError("COMPUTE_FAILED", e.message); }
      const computeMs = now() - t;
      this.stats.processed++;

      const want = (p.flags & P.FLAG_WANT_LOGITS) !== 0;
      await this.sendPacket(P.encodeToken({ sessionId: BigInt(p.sessionId), seq: p.seq, token: r.nextByte, timing: { computeMs, T }, logits: want ? r.logits : null }));
      this._emit("result", { sessionId: p.sessionId, seq: p.seq, token: r.nextByte, computeMs });
    }

    _onDispose() { this.seen.clear(); }

    async _problem(err, sessionId, seq) {
      this.stats.rejected++;
      this._emit("error", { code: err.code, message: err.message, sessionId });
      try { await this.sendPacket(P.encodeError({ sessionId: BigInt(sessionId || 0), seq: seq || 0, code: err.code, message: err.message })); }
      catch (e) { this._emit("error", { code: "SEND_FAILED", message: "could not report the error to Device A: " + e.message }); }
    }
  }

  const api = { PipelineA, PipelineB, modelFingerprint, verifyPartitionBuffer, partitionChecksum, SPLIT_AFTER_LAYER };
  if (isNode) module.exports = api; else root.MeshPipeline = api;
})(typeof window !== "undefined" ? window : globalThis);
