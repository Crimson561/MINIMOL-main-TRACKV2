"use strict";
/* MINIMOL Stage 3: DataChannel test helpers. Loaded after main.js, uses its globals (S, note).
   Console usage once both devices show "Connection successful":
     await meshTest.text()            // ping -> pong round trip
     await meshTest.binary()          // 100 KB ArrayBuffer out, echoed back, bytes compared (default size, or pass bytes)
     await meshTest.all()             // both
     await meshTest.tensor()          // Stage 4: 7 tensors (1-D..4-D, zeros, empty, large) out, echoed back, verified on both devices
   Run them on Device A, then again on Device B to test the reverse direction.
   Test traffic is a string, or an ArrayBuffer starting with the bytes "TEST", so it never collides with MESH packets. */
(() => {
  const HIGH = 1048576, LOW = 262144;          // backpressure: wait above 1 MiB buffered, resume at 256 KiB
  const HDR = 13;                              // binary frame: "TEST"(4) | kind(1) | id(4) | length(4) | payload
  const REQ = 1, ECHO = 2, WAIT_MS = 10000;
  const pending = new Map();
  let nextId = 1;

  const maxSize = () => (S.pc && S.pc.sctp && S.pc.sctp.maxMessageSize) || 65536;
  const sizeOf = (d) => (typeof d === "string" ? new TextEncoder().encode(d).byteLength : d.byteLength);

  // Resolves on `evt`, rejects if the channel closes first or `ms` passes.
  const waitFor = (dc, evt, ms) => new Promise((ok, no) => {
    const done = (fn, v) => { clearTimeout(t); dc.removeEventListener(evt, onEvt); dc.removeEventListener("close", onClose); fn(v); };
    const onEvt = () => done(ok), onClose = () => done(no, new Error("DataChannel closed while waiting"));
    const t = setTimeout(() => done(no, new Error("Timed out waiting for DataChannel " + evt)), ms);
    dc.addEventListener(evt, onEvt); dc.addEventListener("close", onClose);
  });

  async function sendSafe(data) {
    const dc = S.dc;
    if (!dc) throw new Error("No DataChannel yet");
    if (dc.readyState === "connecting") await waitFor(dc, "open", 5000);     // message sent before the channel is ready
    if (dc.readyState !== "open") throw new Error("DataChannel is " + dc.readyState);
    const size = sizeOf(data), max = maxSize();
    if (size > max) throw new Error(`Message is ${size} bytes; DataChannel limit is ${max}`);
    if (dc.bufferedAmount > HIGH) {                                            // backpressure
      dc.bufferedAmountLowThreshold = LOW;
      await waitFor(dc, "bufferedamountlow", 15000);
    }
    dc.send(data); S.bytes += size;
  }

  const pattern = (n, id) => { const u = new Uint8Array(n); for (let i = 0; i < n; i++) u[i] = (i * 31 + id * 7 + (i >> 8)) & 255; return u; };
  const same = (a, b) => { if (a.byteLength !== b.byteLength) return false; for (let i = 0; i < a.byteLength; i++) if (a[i] !== b[i]) return false; return true; };
  const frame = (kind, id, payload) => { const u = new Uint8Array(HDR + payload.byteLength), v = new DataView(u.buffer); u.set([0x54, 0x45, 0x53, 0x54], 0); v.setUint8(4, kind); v.setUint32(5, id); v.setUint32(9, payload.byteLength); u.set(payload, HDR); return u.buffer; };
  const expect = (key, ms = WAIT_MS) => new Promise((ok, no) => {
    const t = setTimeout(() => { pending.delete(key); no(new Error("No reply from the other device")); }, ms);
    pending.set(key, (v) => { clearTimeout(t); pending.delete(key); ok(v); });
  });
  // ---- Stage 4: tensor serialization over the DataChannel (uses MeshProtocol.encodeTensor/decodeTensor) ----
  const T_ECHO = 2;                            // flags bit 1 marks the echo; bit 0 is is_eos in the protocol
  const TENSOR_CASES = [["1-D", [8]], ["2-D", [3, 4]], ["3-D", [2, 3, 4]], ["4-D", [1, 2, 3, 5]], ["zeros", [2, 5]], ["empty", [0]], ["large [1,31,384]", [1, 31, 384]]];
  // Deterministic from the case index alone, so the receiver can rebuild the expected tensor itself.
  // Only exact integer arithmetic and /8 (exact in binary): no Math.sin, which can differ between browser engines.
  function makeTensor(idx) {
    const [name, shape] = TENSOR_CASES[idx], n = shape.reduce((a, b) => a * b, 1), data = new Float32Array(n);
    if (name !== "zeros") { for (let i = 0; i < n; i++) data[i] = (((i * 37 + idx * 11) % 201) - 100) / 8; if (n > 1) data[1] = 0.1; }  // negatives, positives, one non-dyadic value
    return { name, shape, data };
  }
  const sameBits = (a, b) => { if (a.length !== b.length) return false; const x = new Uint32Array(a.buffer, a.byteOffset, a.length), y = new Uint32Array(b.buffer, b.byteOffset, b.length); for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false; return true; };
  const tensorProblem = (got, want) => got.dtype !== "float32" ? "dtype " + got.dtype : JSON.stringify(got.shape) !== JSON.stringify(want.shape) ? "shape " + JSON.stringify(got.shape) + " != " + JSON.stringify(want.shape) : got.numel !== want.data.length ? "element count" : !sameBits(got.data, want.data) ? "values differ" : null;

  async function tensor() {
    const MP = window.MeshProtocol;
    if (!MP || !MP.encodeTensor) throw new Error("MeshProtocol with tensor support is not loaded yet (connect first)");
    const id = nextId++, t0 = performance.now(); let bytes = 0;
    for (let idx = 0; idx < TENSOR_CASES.length; idx++) {
      const sent = makeTensor(idx), reply = expect("x" + id + "." + idx), packet = MP.encodeTensor({ sessionId: id, seq: idx, data: sent.data, shape: sent.shape });
      await sendSafe(packet); bytes += packet.byteLength;
      const problem = tensorProblem(await reply, sent);          // the echo, rebuilt by decodeTensor on THIS device
      if (problem) throw new Error(`Tensor "${sent.name}" came back wrong: ${problem}`);
    }
    return report(`Tensor round trip OK: ${TENSOR_CASES.length} tensors (${bytes} bytes sent) identical both ways (${(performance.now() - t0).toFixed(0)} ms)`);
  }

  function handleTensor(raw) {
    const MP = window.MeshProtocol; let t;
    S.bytes += raw.byteLength;
    try { t = MP.decodeTensor(raw); } catch (e) { note("Tensor REJECTED: " + e.message); return; }   // validation failed: never trust it
    if (t.flags & T_ECHO) { const f = pending.get("x" + t.sessionId + "." + t.seq); if (f) f(t); return; }
    const want = t.seq < TENSOR_CASES.length ? makeTensor(t.seq) : null, problem = want ? tensorProblem(t, want) : "unknown test case " + t.seq;
    note(`Tensor "${want ? want.name : "?"}" received: shape [${t.shape.join(", ")}] ${t.dtype}, ${t.numel} elements, ${problem ? "MISMATCH (" + problem + ")" : "identical to expected"}`);
    sendSafe(MP.encodeTensor({ sessionId: BigInt(t.sessionId), seq: t.seq, flags: T_ECHO, data: t.data, shape: t.shape })).catch((e) => note("Tensor echo failed: " + e.message));
  }

  const report = (r) => { note(r); console.log("[meshTest] " + r); return r; };

  async function text() {
    const id = nextId++, t0 = performance.now(), reply = expect("t" + id);
    await sendSafe("ping:" + id);
    await reply;
    return report(`Text round trip OK (${(performance.now() - t0).toFixed(0)} ms)`);
  }

  async function binary(bytes = 100 * 1024) {
    const id = nextId++, sent = pattern(bytes, id), t0 = performance.now(), reply = expect("b" + id);
    await sendSafe(frame(REQ, id, sent));
    const echoed = await reply;
    if (!same(echoed, sent)) throw new Error("Echoed bytes differ from the original");
    return report(`Binary round trip OK: ${bytes} bytes identical both ways (${(performance.now() - t0).toFixed(0)} ms)`);
  }

  async function all() { const a = await text(), b = await binary(); return [a, b]; }

  // Called from onPacket in main.js; returns true when the message was test traffic.
  function handle(raw) {
    if (typeof raw === "string") {
      const m = /^(ping|pong):(\d+)$/.exec(raw);
      if (!m) { note("Text received: " + raw.slice(0, 80)); return true; }
      if (m[1] === "ping") sendSafe("pong:" + m[2]).catch((e) => note("Reply failed: " + e.message));
      else if (pending.has("t" + m[2])) pending.get("t" + m[2])(true);
      return true;
    }
    if (!(raw instanceof ArrayBuffer) || raw.byteLength < HDR) return false;
    const u = new Uint8Array(raw);
    if (u[0] === 0x4d && u[1] === 0x45 && u[2] === 0x53 && u[3] === 0x48 && u[5] === 0x05) { handleTensor(raw); return true; }   // "MESH" packet of type TENSOR (Stage 4)
    if (u[0] !== 0x54 || u[1] !== 0x45 || u[2] !== 0x53 || u[3] !== 0x54) return false;     // not "TEST": a normal MESH packet
    const v = new DataView(raw), kind = v.getUint8(4), id = v.getUint32(5), len = v.getUint32(9), payload = u.subarray(HDR);
    S.bytes += raw.byteLength;
    if (kind === REQ) {
      const ok = len === payload.byteLength && same(payload, pattern(len, id));
      note(`Binary test received: ${payload.byteLength} bytes, ${ok ? "identical to original" : "MISMATCH"}`);
      sendSafe(frame(ECHO, id, payload)).catch((e) => note("Echo failed: " + e.message));
    } else if (kind === ECHO && pending.has("b" + id)) pending.get("b" + id)(payload.slice());
    return true;
  }

  window.meshTest = { handle, text, binary, tensor, all, sendSafe };
})();
