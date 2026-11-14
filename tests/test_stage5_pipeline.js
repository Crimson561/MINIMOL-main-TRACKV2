/**
 * Stage 5 — real intermediate-activation transfer between Device A and Device B.
 *
 * Standards used (measured in Stage 5 inspection, see TESTING.md):
 *   distributed JS  vs  local JS      -> must be IDENTICAL (same engine, same weights, lossless transfer)
 *   JS              vs  PyTorch       -> within 1e-5 (float32 rounding is ~1e-6; JS and PyTorch are NOT bit-identical)
 *
 * Needs: python model/export_weights.py && python model/export_stage5_reference.py
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const { MeshPartitionA, MeshPartitionB, MeshFullModel } = require("../engine/inference_engine");
const P = require("../protocol/mesh_protocol");
const { PipelineA, PipelineB, modelFingerprint, verifyPartitionBuffer, partitionChecksum } = require("../engine/mesh_pipeline");

const WEIGHTS_DIR = path.join(__dirname, "..", "room-server", "public", "weights");
const FIXTURE = path.join(__dirname, "fixtures", "stage5_reference.json");
const D = 384;

/* ---------------- load weights, fixture, local reference ---------------- */
const ab = (f) => { const b = fs.readFileSync(path.join(WEIGHTS_DIR, f)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
if (!fs.existsSync(path.join(WEIGHTS_DIR, "weights_manifest.json"))) { console.error("Weights missing. Run: python model/export_weights.py"); process.exit(1); }
if (!fs.existsSync(FIXTURE)) { console.error("Fixture missing. Run: python model/export_stage5_reference.py"); process.exit(1); }
const manifest = JSON.parse(fs.readFileSync(path.join(WEIGHTS_DIR, "weights_manifest.json"), "utf-8"));
const bufA = ab("partition_a.bin"), bufB = ab("partition_b.bin"), bufFull = ab("full_model.bin");
const fixture = JSON.parse(fs.readFileSync(FIXTURE, "utf-8"));
const TOL = fixture.tolerance;
const f32 = (b64) => { const b = Buffer.from(b64, "base64"); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
for (const c of fixture.cases) { c.act = f32(c.act_b64); c.logits = f32(c.logits_b64); }

const realA = MeshPartitionA.fromBuffer(bufA, manifest);      // Device A: ONLY partition A weights
const realB = MeshPartitionB.fromBuffer(bufB, manifest);      // Device B: ONLY partition B weights
const local = MeshFullModel.fromBuffer(bufFull, manifest);    // single-device JS reference (full_model.bin)
const FP = modelFingerprint(manifest);

const bitEqual = (a, b) => { if (a.length !== b.length) return false; const x = new Uint32Array(a.buffer, a.byteOffset, a.length), y = new Uint32Array(b.buffer, b.byteOffset, b.length); for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false; return true; };
const valuesEqual = (a, b) => { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; };
const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(cond, ms = 2000) { const t0 = Date.now(); while (!cond()) { if (Date.now() - t0 > ms) throw new Error("condition not met in " + ms + " ms"); await sleep(2); } }
async function rejects(promise, code, extra = {}) {
  try { await promise; } catch (e) { assert.strictEqual(e.code, code, `expected error code ${code}, got ${e.code}: ${e.message}`); for (const [k, v] of Object.entries(extra)) assert.strictEqual(e[k], v); return e; }
  assert.fail(`expected rejection with code ${code}, but it resolved`);
}

/* ---------------- mock DataChannel pair (enforces the message-size limit like a browser does) ---------------- */
class Link {
  constructor(maxMessageSize, latency = 1) { this.max = maxMessageSize; this.latency = latency; this.log = { AtoB: [], BtoA: [] }; this.filter = null; this.biggest = 0; this.toA = null; this.toB = null; }
  transport(dir) {
    return {
      maxMessageSize: () => this.max,
      send: (buf) => {
        if (buf.byteLength > this.max) throw new RangeError(`message of ${buf.byteLength} bytes exceeds DataChannel limit ${this.max}`);   // a browser would close the channel
        this.biggest = Math.max(this.biggest, buf.byteLength);
        this.log[dir].push(buf);
        let out = [buf.slice(0)];
        if (this.filter) out = [].concat(this.filter(dir, buf.slice(0), this.log[dir].length - 1)).filter(Boolean);
        for (const m of out) setTimeout(() => (dir === "AtoB" ? this.toB : this.toA)(m), this.latency);
      },
    };
  }
}
function spy(engine) {
  return { calls: 0, received: null, untouched: null, sent: null,
    forward(x, T) {
      this.calls++;
      if (T === undefined) { const out = engine.forward(x); this.sent = out.slice(); return out; }   // partition A: tokens in
      this.received = x.slice();                                                                         // partition B: activation in
      const r = engine.forward(x, T);
      this.untouched = bitEqual(x, this.received);                                                       // B must never write into the received buffer
      return r;
    } };
}
function nodes({ max = 262144, timeoutMs = 5000, engineA = realA, engineB = realB, fpA = FP, fpB = FP, latency = 1 } = {}) {
  const link = new Link(max, latency), evA = [], evB = [];
  const pa = new PipelineA({ engine: engineA, fingerprint: fpA, transport: link.transport("AtoB"), timeoutMs, onEvent: (t, i) => evA.push({ t, ...i }) });
  const pb = new PipelineB({ engine: engineB, fingerprint: fpB, transport: link.transport("BtoA"), timeoutMs, onEvent: (t, i) => evB.push({ t, ...i }) });
  link.toA = (m) => pa.handleMessage(m); link.toB = (m) => pb.handleMessage(m);
  return { link, pa, pb, evA, evB };
}
async function connected(opts) { const n = nodes(opts); await n.pa.sendHello(); await n.pb.sendHello(); await until(() => n.pa.peer && n.pb.peer); return n; }

let passed = 0, failed = 0;
async function test(name, fn) {
  try { await fn(); passed++; console.log("  ✓ " + name); }
  catch (e) { failed++; console.log("  ✗ " + name + "\n      " + String(e && e.stack || e).split("\n").slice(0, 4).join("\n      ")); }
}
const section = (s) => console.log("\n" + s);

(async () => {
  console.log("======================================================================");
  console.log("STAGE 5 — ACTIVATION TRANSFER BETWEEN DEVICE A AND DEVICE B");
  console.log("======================================================================");

  /* ===== 1. weights consistency ===== */
  section("1. Weights, fixture and partitions agree");
  await test("partition files match the manifest and the PyTorch fixture checksums", () => {
    assert.strictEqual(partitionChecksum(bufA), manifest.partition_a.checksum_adler32);
    assert.strictEqual(partitionChecksum(bufB), manifest.partition_b.checksum_adler32);
    assert.strictEqual(fixture.partition_a_adler32, manifest.partition_a.checksum_adler32, "fixture was made from different weights: re-run export_stage5_reference.py");
    assert.strictEqual(fixture.partition_b_adler32, manifest.partition_b.checksum_adler32, "fixture was made from different weights: re-run export_stage5_reference.py");
    verifyPartitionBuffer(bufA, manifest, "a"); verifyPartitionBuffer(bufB, manifest, "b");
  });
  await test("a corrupted or truncated partition download is refused", () => {
    const bad = bufA.slice(0); new Uint8Array(bad)[1000] ^= 1;
    assert.throws(() => verifyPartitionBuffer(bad, manifest, "a"), (e) => e.code === "MODEL_MISMATCH");
    assert.throws(() => verifyPartitionBuffer(bufA.slice(0, bufA.byteLength - 4), manifest, "a"), (e) => e.code === "MODEL_MISMATCH");
    assert.throws(() => verifyPartitionBuffer(bufB, manifest, "a"), (e) => e.code === "MODEL_MISMATCH");   // B's file offered as A's
  });
  await test("each device holds only its own weights (B cannot repeat A's work)", () => {
    const a = Object.keys(realA.weights), b = Object.keys(realB.weights);
    assert(a.includes("wte.weight") && a.includes("h.2.mlp.c_proj.weight"));
    assert(!a.some((k) => /^h\.[345]\./.test(k)) && !a.includes("lm_head.weight") && !a.includes("ln_f.weight"));
    assert(!b.includes("wte.weight") && !b.includes("wpe.weight") && !b.some((k) => /^h\.[012]\./.test(k)));
    assert(b.includes("h.3.ln_1.weight") && b.includes("lm_head.weight"));
  });

  /* ===== 2. Device A partition, locally ===== */
  section("2. Device A's activation (run locally, compared with PyTorch)");
  for (const c of fixture.cases) {
    await test(`${c.name} (T=${c.T}): activation is [1,${c.T},${D}], finite, within ${TOL} of PyTorch`, () => {
      const act = realA.forward(c.ids);
      assert(act instanceof Float32Array);
      assert.strictEqual(act.length, c.T * D);
      assert.deepStrictEqual(c.act_shape, [1, c.T, D]);
      for (let i = 0; i < act.length; i++) assert(Number.isFinite(act[i]));
      const d = maxAbs(act, c.act);
      assert(d <= TOL, `activation differs from PyTorch by ${d}`);
      c.localActDiff = d;
    });
  }
  console.log("      max |JS - PyTorch| on activations: " + Math.max(...fixture.cases.map((c) => c.localActDiff || 0)).toExponential(2));

  /* ===== 3. serialization ===== */
  section("3. Serialize, reconstruct, verify shape / dtype / values");
  for (const c of fixture.cases) {
    await test(`${c.name}: ACTIVATION packet round-trips bit-identically`, () => {
      const act = realA.forward(c.ids);
      const pkt = P.encodeActivation({ sessionId: 123456789012345n, seq: 1, activations: act, shape: [1, c.T, D] });
      const p = P.decodePacket(pkt);
      assert.deepStrictEqual(p.shape, [1, c.T, D]); assert.strictEqual(p.dataType, P.DATA_TYPES.FP32);
      assert.strictEqual(p.sessionId, "123456789012345"); assert.strictEqual(p.seq, 1);
      assert.strictEqual(P.validateActivation(p).T, c.T);
      assert(bitEqual(p.payload, act), "payload differs from the activation that was sent");
    });
  }
  await test("chunking: any message size reassembles to the exact original bytes", () => {
    const act = realA.forward(fixture.cases[5].ids);   // 200 tokens, 307 KB
    const pkt = P.encodeActivation({ sessionId: 9n, seq: 1, activations: act, shape: [1, 200, D] });
    for (const max of [256, 1000, 16384, 65536, 262144, pkt.byteLength - 1, pkt.byteLength]) {
      const frames = P.splitIntoChunks(pkt, max, 5);
      for (const f of frames) assert(f.byteLength <= max, `frame of ${f.byteLength} > ${max}`);
      assert.strictEqual(frames.length > 1, pkt.byteLength > max);
      const r = new P.ChunkReassembler(); let out = null;
      for (const f of frames) out = frames.length === 1 ? f : r.push(f);
      assert(Buffer.from(out).equals(Buffer.from(pkt)), `reassembled bytes differ at max=${max}`);
    }
  });

  /* ===== 4. validation refuses bad data ===== */
  section("4. Device B refuses incomplete or invalid activations");
  const okAct = new Float32Array(2 * D).fill(0.25);
  const mk = (over = {}) => { const o = { sessionId: 1n, seq: 1, activations: okAct, shape: [1, 2, D], ...over }; return P.decodePacket(P.encodeActivation(o)); };
  const bad = (name, p, code) => test(name, () => assert.throws(() => P.validateActivation(p), (e) => e.code === code, `expected ${code}`));
  await bad("rank 2 shape [T,384]", mk({ shape: [2, D] }), "BAD_SHAPE");
  await bad("batch size 2", mk({ shape: [2, 1, D] }), "BAD_SHAPE");
  await bad("hidden size 383", mk({ activations: new Float32Array(383), shape: [1, 1, 383] }), "BAD_SHAPE");
  await bad("sequence length 513", mk({ activations: new Float32Array(513 * D), shape: [1, 513, D] }), "BAD_SHAPE");
  await bad("sequence length 0", mk({ activations: new Float32Array(0), shape: [1, 0, D] }), "BAD_SHAPE");
  await bad("header says T=5 but only 2 rows of data", mk({ shape: [1, 5, D] }), "BAD_SHAPE");
  await bad("NaN inside the activation", mk({ activations: Float32Array.from({ length: 2 * D }, (_, i) => (i === 700 ? NaN : 1)) }), "NON_FINITE");
  await bad("Infinity inside the activation", mk({ activations: Float32Array.from({ length: 2 * D }, (_, i) => (i === 3 ? Infinity : 1)) }), "NON_FINITE");
  await bad("a TOKEN packet where an activation is expected", P.decodePacket(P.encodeToken({ sessionId: 1n, seq: 1, token: 5 })), "BAD_PACKET");
  await test("wrong dtype byte", () => { const b = P.encodeActivation({ sessionId: 1n, seq: 1, activations: okAct, shape: [1, 2, D] }); new DataView(b).setUint8(20, P.DATA_TYPES.UINT8); assert.throws(() => P.validateActivation(P.decodePacket(b)), (e) => e.code === "BAD_DTYPE"); });
  await test("a flipped payload bit fails the checksum", () => { const b = P.encodeActivation({ sessionId: 1n, seq: 1, activations: okAct, shape: [1, 2, D] }); new Uint8Array(b)[200] ^= 4; assert.throws(() => P.decodePacket(b), /checksum/i); });
  await test("chunk protocol rejects gaps, mismatches and oversize declarations", () => {
    const pkt = new ArrayBuffer(5000), fr = P.splitIntoChunks(pkt, 1000, 1);
    assert(fr.length >= 5);
    const code = (fn) => { try { fn(); } catch (e) { return e.code; } return null; };
    let r = new P.ChunkReassembler(); r.push(fr[0]); assert.strictEqual(code(() => r.push(fr[2])), "BAD_CHUNK", "gap");
    r = new P.ChunkReassembler(); assert.strictEqual(code(() => r.push(fr[1])), "BAD_CHUNK", "middle chunk with no start");
    r = new P.ChunkReassembler(); r.push(fr[0]); r.push(fr[1]); assert.strictEqual(code(() => r.push(fr[1])), "BAD_CHUNK", "repeated chunk");
    r = new P.ChunkReassembler({ maxBytes: 4000 }); assert.strictEqual(code(() => r.push(fr[0])), "BAD_CHUNK", "declared size over the cap");
    const other = P.splitIntoChunks(new ArrayBuffer(5000), 1000, 2);
    r = new P.ChunkReassembler(); r.push(fr[0]); assert.strictEqual(code(() => r.push(other[1])), "BAD_CHUNK", "chunk from another message");
    const lying = fr[fr.length - 1].slice(0); new DataView(lying).setUint32(16, 4999, false);
    r = new P.ChunkReassembler(); for (let i = 0; i < fr.length - 1; i++) r.push(fr[i]); assert.strictEqual(code(() => r.push(lying)), "BAD_CHUNK", "size mismatch");
    assert.throws(() => P.splitIntoChunks(pkt, 100, 1), RangeError);
  });

  /* ===== 5. end-to-end over the two-node pipeline ===== */
  section("5. End to end: A -> real activation -> B -> result, compared with local inference and PyTorch");
  for (const max of [262144, 65536, 16384]) {
    await test(`all fixture inputs, DataChannel limit ${max} bytes`, async () => {
      const sA = spy(realA), sB = spy(realB);
      const n = await connected({ max, engineA: sA, engineB: sB });
      let requests = 0, chunkedSeen = false, worstPy = 0;
      for (const c of fixture.cases) {
        const r = await n.pa.infer(c.ids, { wantLogits: true }); requests++;
        const localAct = local.partitionA.forward(c.ids), localOut = local.partitionB.forward(localAct, c.T);
        assert(bitEqual(sA.sent, localAct), `${c.name}: A's activation differs from local`);
        assert(bitEqual(sB.received, localAct), `${c.name}: the activation B received differs from the one A computed`);     // transfer is lossless
        assert.strictEqual(sB.untouched, true, "B modified the received activation buffer");
        assert.strictEqual(r.token, localOut.nextByte, `${c.name}: token differs from local inference`);
        assert(valuesEqual(r.logits, localOut.logits), `${c.name}: distributed logits differ from local inference`);        // identical, not merely close
        assert.strictEqual(r.token, c.argmax, `${c.name}: token differs from PyTorch`);
        const d = maxAbs(r.logits, c.logits); worstPy = Math.max(worstPy, d);
        assert(d <= TOL, `${c.name}: logits differ from PyTorch by ${d}`);
        assert.strictEqual(r.T, c.T); assert.strictEqual(r.activationBytes, 48 + c.T * D * 4);
        if (r.activationBytes > max) { chunkedSeen = true; assert(r.frames > 1); } else assert.strictEqual(r.frames, 1);
      }
      assert(n.link.biggest <= max, "a message exceeded the channel limit");
      assert.strictEqual(sB.calls, requests, "B must compute each activation exactly once");
      assert.strictEqual(n.pb.stats.processed, requests);
      if (max < 262144) assert(chunkedSeen, "expected chunking to be exercised");
      console.log(`      limit ${max}: ${requests} requests, largest message ${n.link.biggest} B, worst |logit - PyTorch| = ${worstPy.toExponential(2)}`);
    });
  }
  await test("full 512-token context (786,480-byte activation) split over a 256 KiB channel", async () => {
    const ids = Array.from({ length: 512 }, (_, i) => (i * 131 + 7) % 256);
    const n = await connected({ max: 262144, timeoutMs: 60000 });
    const r = await n.pa.infer(ids, { wantLogits: true });
    const act = local.partitionA.forward(ids), out = local.partitionB.forward(act, 512);
    assert.strictEqual(r.activationBytes, 48 + 512 * D * 4); assert.strictEqual(r.frames, 4);
    assert.strictEqual(r.token, out.nextByte); assert(valuesEqual(r.logits, out.logits));
  });
  await test("the same prompt twice gives two sessions and identical results", async () => {
    const n = await connected(); const ids = fixture.cases[0].ids;
    const a = await n.pa.infer(ids, { wantLogits: true }), b = await n.pa.infer(ids, { wantLogits: true });
    assert.notStrictEqual(a.sessionId, b.sessionId); assert(valuesEqual(a.logits, b.logits));
    assert.strictEqual(n.pb.stats.processed, 2);
  });
  await test("without wantLogits the reply stays small (token only)", async () => {
    const n = await connected(); const r = await n.pa.infer(fixture.cases[1].ids);
    assert.strictEqual(r.logits, null); assert.strictEqual(r.token, fixture.cases[1].argmax);
    assert(n.link.log.BtoA[n.link.log.BtoA.length - 1].byteLength < 400);
  });

  /* ===== 6. coordination and failures ===== */
  section("6. Coordination: wrong device, wrong request, duplicates, corruption, failures");
  await test("a second request while one is in flight is refused; the first still completes", async () => {
    const n = await connected(); const ids = fixture.cases[0].ids;
    const first = n.pa.infer(ids); await rejects(n.pa.infer(ids), "BUSY"); assert.strictEqual((await first).token, fixture.cases[0].argmax);
  });
  await test("invalid prompts are refused before anything is sent", async () => {
    const n = await connected(); const before = n.link.log.AtoB.length;
    await rejects(n.pa.infer([]), "BAD_INPUT"); await rejects(n.pa.infer(new Array(513).fill(1)), "BAD_INPUT");
    await rejects(n.pa.infer([1, 256]), "BAD_INPUT"); await rejects(n.pa.infer([1.5]), "BAD_INPUT");
    assert.strictEqual(n.link.log.AtoB.length, before);
  });
  await test("B without a loaded partition: A refuses to start (no silent fallback)", async () => {
    const n = await connected({ engineB: null, timeoutMs: 300 });
    assert.strictEqual(n.pa.peer.ok, false); await rejects(n.pa.infer(fixture.cases[0].ids), "NOT_READY");
  });
  await test("an activation sent to a B with no partition gets an ERROR back, not a hang", async () => {
    const n = await connected({ engineB: null, timeoutMs: 300 });
    n.link.transport("AtoB").send(P.encodeActivation({ sessionId: 5n, seq: 1, activations: okAct, shape: [1, 2, D] }));
    await until(() => n.evA.some((e) => e.t === "peer-error")); assert.strictEqual(n.evA.find((e) => e.t === "peer-error").code, "NOT_READY");
  });
  await test("devices with different weights/model never exchange activations", async () => {
    const sB = spy(realB), n = await connected({ fpB: FP.replace(/b\d+$/, "b1"), engineB: sB, timeoutMs: 300 });
    await rejects(n.pa.infer(fixture.cases[0].ids), "MODEL_MISMATCH"); assert.strictEqual(sB.calls, 0); assert.strictEqual(n.link.log.AtoB.length, 1, "only HELLO may be sent");
  });
  await test("two devices claiming the same role are refused", async () => {
    const n = nodes({ timeoutMs: 300 });
    await n.pb.handleMessage(P.encodeControl({ sessionId: 0, command: "HELLO", data: { role: "B", loaded: true, fingerprint: FP } }));   // B hears another "B"
    assert.strictEqual(n.pb.peer.ok, false); assert.strictEqual(n.pb.peer.fatal, true);
    await rejects(n.pb.waitReady(100), "MODEL_MISMATCH");
  });
  await test("a corrupted activation (single). B reports CHECKSUM, A fails fast, B computes nothing", async () => {
    for (const max of [262144, 16384]) {
      const sB = spy(realB), n = await connected({ max, engineB: sB, timeoutMs: 5000 }); let hit = false;
      n.link.filter = (dir, buf) => { if (dir === "AtoB" && !hit && buf.byteLength > 1000) { hit = true; new Uint8Array(buf)[Math.min(300, buf.byteLength - 1)] ^= 0x10; } return buf; };
      const t0 = Date.now(); const e = await rejects(n.pa.infer(fixture.cases[1].ids), "CHECKSUM", { fromPeer: true });
      assert(Date.now() - t0 < 1500, "should fail immediately, not by timeout"); assert.strictEqual(sB.calls, 0);
      n.link.filter = null; assert.strictEqual((await n.pa.infer(fixture.cases[1].ids)).token, fixture.cases[1].argmax, "pipeline recovers for the next request");
    }
  });
  await test("a lost middle chunk is detected immediately", async () => {
    const sB = spy(realB), n = await connected({ max: 16384, engineB: sB }); let i = 0;
    n.link.filter = (dir, buf) => (dir === "AtoB" && buf.byteLength > 1000 && ++i === 2 ? null : buf);
    await rejects(n.pa.infer(fixture.cases[4].ids), "BAD_CHUNK", { fromPeer: true }); assert.strictEqual(sB.calls, 0);
  });
  await test("a lost final chunk times out, then the next request still works", async () => {
    const sB = spy(realB), n = await connected({ max: 16384, engineB: sB, timeoutMs: 400 }); let drop = true;
    n.link.filter = (dir, buf) => { if (dir === "AtoB" && drop && P.isChunk(buf) && new DataView(buf).getUint32(8) === new DataView(buf).getUint32(12) - 1) return null; return buf; };
    await rejects(n.pa.infer(fixture.cases[4].ids), "TIMEOUT"); assert.strictEqual(sB.calls, 0); drop = false;
    assert.strictEqual((await n.pa.infer(fixture.cases[4].ids, { timeoutMs: 30000 })).token, fixture.cases[4].argmax);
    assert(n.evB.some((e) => e.t === "warn"), "B should have noted the discarded partial message");
  });
  await test("replaying an already-processed activation is refused; B computes once", async () => {
    const sB = spy(realB), n = await connected({ engineB: sB }); const ids = fixture.cases[2].ids;
    const r = await n.pa.infer(ids); assert.strictEqual(r.token, fixture.cases[2].argmax);
    const recorded = n.link.log.AtoB.find((b) => P.decodePacket(b).msgType === P.MSG_TYPES.ACTIVATION);
    await n.pb.handleMessage(recorded.slice(0)); await n.pb.handleMessage(recorded.slice(0));
    assert.strictEqual(sB.calls, 1); assert.strictEqual(n.pb.stats.processed, 1);
    assert.strictEqual(n.evB.filter((e) => e.t === "error" && e.code === "DUPLICATE").length, 2);
  });
  await test("replies for another session or sequence are ignored; the right one completes", async () => {
    const n = await connected(); const ids = fixture.cases[3].ids;
    const p = n.pa.infer(ids, { sessionId: 7777n });
    await n.pa.handleMessage(P.encodeToken({ sessionId: 9999n, seq: 1, token: 1 }));       // other session
    await n.pa.handleMessage(P.encodeToken({ sessionId: 7777n, seq: 2, token: 1 }));       // right session, wrong seq
    await n.pa.handleMessage(P.encodeError({ sessionId: 9999n, seq: 1, code: "X", message: "not for you" }));
    const r = await p; assert.strictEqual(r.token, fixture.cases[3].argmax); assert.strictEqual(r.sessionId, "7777");
    assert.strictEqual(n.evA.filter((e) => e.t === "warn").length, 2);
  });
  await test("a reply whose token is not the argmax of its logits is rejected", async () => {
    const n = await connected(); const p = n.pa.infer(fixture.cases[0].ids, { sessionId: 4242n, wantLogits: true });
    const lg = new Float32Array(256); lg[10] = 1;
    await n.pa.handleMessage(P.encodeToken({ sessionId: 4242n, seq: 1, token: 99, logits: lg })); await rejects(p, "BAD_REPLY");
  });
  await test("B reports a bad shape / NaN / oversize activation as ERROR and computes nothing", async () => {
    const sB = spy(realB), n = await connected({ engineB: sB, max: 1 << 20 });
    const send = (o) => n.link.transport("AtoB").send(P.encodeActivation({ sessionId: 31n, seq: 1, ...o }));
    send({ activations: okAct, shape: [1, 5, D] }); send({ activations: Float32Array.from({ length: 2 * D }, () => NaN), shape: [1, 2, D] });
    send({ activations: new Float32Array(513 * D), shape: [1, 513, D] });
    await until(() => n.evA.filter((e) => e.t === "peer-error").length === 3);
    assert.deepStrictEqual(n.evA.filter((e) => e.t === "peer-error").map((e) => e.code), ["BAD_SHAPE", "NON_FINITE", "BAD_SHAPE"]); assert.strictEqual(sB.calls, 0);
  });
  await test("if B's compute throws, A gets COMPUTE_FAILED immediately", async () => {
    const n = await connected({ engineB: { forward() { throw new Error("boom"); } } });
    await rejects(n.pa.infer(fixture.cases[0].ids), "COMPUTE_FAILED", { fromPeer: true });
  });
  await test("a failed transfer is never silent (send error -> rejection, pending slot released)", async () => {
    const n = await connected({ max: 16384 });
    const honest = n.pa.transport.maxMessageSize;
    n.pa.transport.maxMessageSize = () => 1 << 30;      // a sender that does not chunk: the channel refuses the oversized message, as a browser would
    const c = fixture.cases[2]; let err = null;      // 13 tokens = 20,016 bytes > 16,384
    try { await n.pa.infer(c.ids); } catch (e) { err = e; }
    assert(err instanceof RangeError, "expected the send failure to surface, got " + err);
    n.pa.transport.maxMessageSize = honest;
    assert.strictEqual((await n.pa.infer(c.ids)).token, c.argmax);
  });
  await test("B silent: A times out, the pending slot is released, the next request works", async () => {
    const n = await connected({ timeoutMs: 300 }); let mute = true;
    n.link.filter = (dir, buf) => (dir === "AtoB" && mute ? null : buf);
    await rejects(n.pa.infer(fixture.cases[0].ids), "TIMEOUT"); mute = false;
    assert.strictEqual((await n.pa.infer(fixture.cases[0].ids)).token, fixture.cases[0].argmax);
  });

  console.log("\n======================================================================");
  console.log(failed ? `STAGE 5 TESTS FAILED: ${passed} passed, ${failed} failed` : `STAGE 5 TESTS PASSED: ${passed} passed, 0 failed`);
  console.log("======================================================================");
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
