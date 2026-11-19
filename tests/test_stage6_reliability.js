/**
 * Stage 6 — reliability: failure handling, cleanup and recovery.
 *
 * WHAT KIND OF TESTS THESE ARE (read this before trusting a green result):
 *   [Unit]            pure logic: the connection state machine, reconnect backoff, the chunk reassembler.
 *   [Node simulation] the REAL PipelineA / PipelineB / protocol code, connected by an in-memory fake DataChannel that WE make
 *                     misbehave (close, drop, delay, duplicate, corrupt). Deterministic and fast.
 *   NOT covered here (see docs/history/STAGE6_STATUS.md): real WebRTC / Socket.IO / browser behaviour (scripted two-page browser run) and
 *   physical devices. A fake channel only proves that OUR code reacts correctly to the failures WE imagined. It does not prove that
 *   a real network fails in those ways, or that browsers report failures at the moments we assume.
 *
 * Needs: python model/export_weights.py && python model/export_stage5_reference.py
 */
"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const { MeshPartitionA, MeshPartitionB, MeshFullModel } = require("../engine/inference_engine");
const P = require("../protocol/mesh_protocol");
const { PipelineA, PipelineB, modelFingerprint } = require("../engine/mesh_pipeline");
const CS = require("../Frontend/connection_state");

const WEIGHTS_DIR = path.join(__dirname, "..", "room-server", "public", "weights");
const FIXTURE = path.join(__dirname, "fixtures", "stage5_reference.json");
if (!fs.existsSync(path.join(WEIGHTS_DIR, "weights_manifest.json"))) { console.error("Weights missing. Run: python model/export_weights.py"); process.exit(1); }
if (!fs.existsSync(FIXTURE)) { console.error("Fixture missing. Run: python model/export_stage5_reference.py"); process.exit(1); }
const ab = (f) => { const b = fs.readFileSync(path.join(WEIGHTS_DIR, f)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const manifest = JSON.parse(fs.readFileSync(path.join(WEIGHTS_DIR, "weights_manifest.json"), "utf-8"));
const fixture = JSON.parse(fs.readFileSync(FIXTURE, "utf-8"));
const realA = MeshPartitionA.fromBuffer(ab("partition_a.bin"), manifest);
const realB = MeshPartitionB.fromBuffer(ab("partition_b.bin"), manifest);
const local = MeshFullModel.fromBuffer(ab("full_model.bin"), manifest);
const FP = modelFingerprint(manifest);
const D = 384, C = fixture.cases;

/* ---------- helpers ---------- */
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(cond, ms = 2000) { const t0 = Date.now(); while (!cond()) { if (Date.now() - t0 > ms) throw new Error("condition not met in " + ms + " ms"); await sleep(2); } }
async function rejects(promise, code, extra = {}) {
  try { await promise; } catch (e) { assert.strictEqual(e.code, code, `expected error code ${code}, got ${e.code}: ${e.message}`); for (const [k, v] of Object.entries(extra)) assert.strictEqual(e[k], v, `error.${k}`); return e; }
  assert.fail(`expected rejection with code ${code}, but it resolved`);
}
const valuesEqual = (a, b) => { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; };

// Count every timer created while a test runs, so a test can assert "nothing is left running" (the timer-leak check).
const realSet = global.setTimeout, realClear = global.clearTimeout, live = new Map();
global.setTimeout = (fn, ms, ...a) => { const t = realSet((...x) => { live.delete(t); fn(...x); }, ms, ...a); live.set(t, ms || 0); return t; };
global.clearTimeout = (t) => { live.delete(t); return realClear(t); };
const longTimers = (minMs = 100) => [...live.values()].filter((ms) => ms >= minMs).length;   // ignore the 1-2 ms link-latency timers

let unhandled = 0; process.on("unhandledRejection", (e) => { unhandled++; console.log("      UNHANDLED REJECTION: " + (e && e.message)); });

/** In-memory DataChannel pair. Like a browser it enforces the message limit; unlike a browser it can be told to fail. */
class Link {
  constructor(maxMessageSize, latency = 1) { this.max = maxMessageSize; this.latency = latency; this.log = { AtoB: [], BtoA: [] }; this.filter = null; this.closed = false; this.broken = false; this.failAfter = null; this.sentCount = 0; this.toA = null; this.toB = null; }
  close() { this.closed = true; }
  transport(dir) {
    return {
      maxMessageSize: () => this.max,
      send: (buf) => {
        if (this.closed || this.broken) throw new P.MeshError("CHANNEL_CLOSED", "DataChannel is closed");
        if (this.failAfter !== null && this.sentCount >= this.failAfter) { this.broken = true; throw new P.MeshError("CHANNEL_CLOSED", "DataChannel closed during transmission"); }   // frames already sent are still delivered, like a real channel
        if (buf.byteLength > this.max) throw new RangeError(`message of ${buf.byteLength} bytes exceeds DataChannel limit ${this.max}`);
        this.sentCount++; this.log[dir].push(buf);
        let out = [buf.slice(0)];
        if (this.filter) out = [].concat(this.filter(dir, buf.slice(0), this.log[dir].length - 1)).filter(Boolean);
        for (const m of out) setTimeout(() => { if (!this.closed) (dir === "AtoB" ? this.toB : this.toA)(m); }, this.latency);
      },
    };
  }
}
function nodes({ max = 262144, timeoutMs = 5000, reassemblyTimeoutMs = 10000, engineA = realA, engineB = realB, fpA = FP, fpB = FP } = {}) {
  const link = new Link(max), evA = [], evB = [];
  const pa = new PipelineA({ engine: engineA, fingerprint: fpA, transport: link.transport("AtoB"), timeoutMs, reassemblyTimeoutMs, onEvent: (t, i) => evA.push({ t, ...i }) });
  const pb = new PipelineB({ engine: engineB, fingerprint: fpB, transport: link.transport("BtoA"), timeoutMs, reassemblyTimeoutMs, onEvent: (t, i) => evB.push({ t, ...i }) });
  link.toA = (m) => pa.handleMessage(m); link.toB = (m) => pb.handleMessage(m);
  return { link, pa, pb, evA, evB };
}
async function connected(opts) { const n = nodes(opts); await n.pa.sendHello(); await n.pb.sendHello(); await until(() => n.pa.peer && n.pb.peer); return n; }
function spy(engine, hook) { return { calls: 0, forward(x, T) { this.calls++; if (hook) hook(this); return T === undefined ? engine.forward(x) : engine.forward(x, T); } }; }
const msgType = (buf) => P.decodePacket(buf).msgType;

let passed = 0, failed = 0;
async function test(name, fn) {
  const before = unhandled;
  try { await fn(); assert.strictEqual(unhandled, before, "an unhandled promise rejection happened during this test"); passed++; console.log("  ✓ " + name); }
  catch (e) { failed++; console.log("  ✗ " + name + "\n      " + String(e && e.stack || e).split("\n").slice(0, 4).join("\n      ")); }
}
const section = (s) => console.log("\n" + s);

(async () => {
  console.log("======================================================================");
  console.log("STAGE 6 — RELIABILITY, ERROR HANDLING, CLEANUP, RECOVERY");
  console.log("======================================================================");
  assert.notStrictEqual(C[0].argmax, C[1].argmax); assert.notStrictEqual(C[2].argmax, C[3].argmax);   // tests below tell replies apart by their token

  /* ================= [Unit] state machine ================= */
  section("[Unit] Connection state machine (Frontend/connection_state.js)");
  await test("every state can be RESET to DISCONNECTED; FAILED has no other way out", () => {
    for (const s of CS.STATES) assert.strictEqual(CS.nextState(s, "RESET"), "DISCONNECTED", s);
    assert.deepStrictEqual(Object.keys(CS.TRANSITIONS.FAILED), ["RESET"]);
  });
  await test("INFERENCE_READY is reachable only through CONNECTED (a dead link can never look 'ready')", () => {
    for (const [s, row] of Object.entries(CS.TRANSITIONS)) for (const [ev, to] of Object.entries(row)) if (to === "INFERENCE_READY") assert.strictEqual(s, "CONNECTED", `${s} --${ev}--> INFERENCE_READY`);
    assert.strictEqual(CS.nextState("FAILED", "PIPELINE_READY"), null); assert.strictEqual(CS.nextState("DISCONNECTED", "DC_OPEN"), null);
  });
  await test("every state is reachable from DISCONNECTED, and every failure ends somewhere defined", () => {
    const seen = new Set(["DISCONNECTED"]), todo = ["DISCONNECTED"];
    while (todo.length) for (const to of Object.values(CS.TRANSITIONS[todo.pop()])) if (!seen.has(to)) { seen.add(to); todo.push(to); }
    assert.deepStrictEqual([...seen].sort(), [...CS.STATES].sort());
    for (const s of ["CONNECTING", "CONNECTED", "INFERENCE_READY", "RECONNECTING"]) assert.strictEqual(CS.nextState(s, "FAIL"), "FAILED", s);
  });
  await test("a refused event changes nothing; stale duplicates are harmless", () => {
    const seen = []; const m = CS.createConnectionMachine({ onChange: (to, from, ev) => seen.push(`${from}>${to}`) });
    assert.strictEqual(m.send("DC_OPEN"), false); assert.strictEqual(m.state, "DISCONNECTED");           // not started yet
    assert.strictEqual(m.send("START"), true); assert.strictEqual(m.send("START"), false);               // duplicate START refused
    assert.strictEqual(m.send("DC_OPEN"), true); assert.strictEqual(m.send("PIPELINE_READY"), true); assert.strictEqual(m.state, "INFERENCE_READY");
    assert.strictEqual(m.send("LINK_LOST"), true); assert.strictEqual(m.send("LINK_LOST"), false);       // second report of the same loss
    assert.strictEqual(m.state, "RECONNECTING"); assert.strictEqual(m.send("PIPELINE_READY"), false);
    assert.deepStrictEqual(seen, ["DISCONNECTED>CONNECTING", "CONNECTING>CONNECTED", "CONNECTED>INFERENCE_READY", "INFERENCE_READY>RECONNECTING"]);
  });
  await test("reconnection is BOUNDED: exactly maxReconnects attempts, backoff doubles, then it stops", () => {
    const plan = (attempts) => CS.reconnectPlan({ attempts, max: 2, baseMs: 1000 });
    assert.deepStrictEqual(plan(0), { retry: true, attempt: 1, delayMs: 1000 }); assert.deepStrictEqual(plan(1), { retry: true, attempt: 2, delayMs: 2000 });
    assert.strictEqual(plan(2).retry, false); assert.strictEqual(plan(99).retry, false);
    let n = 0, attempts = 0; while (CS.reconnectPlan({ attempts, max: 2, baseMs: 1 }).retry) { attempts++; n++; if (n > 100) assert.fail("unbounded"); } assert.strictEqual(n, 2);
  });

  /* ================= [Unit] reassembler ================= */
  section("[Unit] Chunk reassembler (protocol/mesh_protocol.js)");
  await test("after a bad chunk, the rest of that message is ignored (no error storm); a new message works", () => {
    const frames = P.splitIntoChunks(new ArrayBuffer(5000), 1000, 7), next = P.splitIntoChunks(new ArrayBuffer(3000), 1000, 8);
    const r = new P.ChunkReassembler(); r.push(frames[0]);
    assert.throws(() => r.push(frames[2]), (e) => e.code === "BAD_CHUNK");            // gap: frame 1 was lost
    for (let i = 3; i < frames.length; i++) assert.strictEqual(r.push(frames[i]), null);  // stragglers: silently dropped, not 3 more errors
    assert.strictEqual(r.suppressed, frames.length - 3); assert.strictEqual(r.inProgress, false);
    let out = null; for (const f of next) out = r.push(f); assert.strictEqual(out.byteLength, 3000);
  });
  await test("reset() forgets both a partial message and the ignore-list", () => {
    const f = P.splitIntoChunks(new ArrayBuffer(5000), 1000, 9), r = new P.ChunkReassembler();
    r.push(f[0]); r.reset(); assert.strictEqual(r.inProgress, false);
    assert.throws(() => r.push(f[1]), (e) => e.code === "BAD_CHUNK");                  // frame 1 with no message in progress: error AND now poisoned
    r.reset(); assert.throws(() => r.push(f[2]), (e) => e.code === "BAD_CHUNK");      // after reset() it is judged afresh, not silently ignored
  });

  /* ================= [Node simulation] the ten required failure scenarios ================= */
  section("[Node simulation] 1. DataChannel closes during inference");
  await test("channel closes while A waits for B's reply: the request fails in well under the timeout and leaves nothing behind", async () => {
    const t0long = longTimers(1000);
    const n = await connected({ timeoutMs: 5000 }); n.link.filter = (dir, buf) => (dir === "AtoB" ? null : buf);   // B never receives the activation
    const p = n.pa.infer(C[1].ids).catch((e) => e); await sleep(80); assert.notStrictEqual(n.pa.pending, null);
    const t = Date.now(); n.link.close(); n.pa.dispose("DataChannel closed");                                       // exactly what main.js closePeer() does
    const e = await p; assert.strictEqual(e.code, "CHANNEL_CLOSED"); assert(Date.now() - t < 300, "must fail at once, not after the 5 s timeout");
    assert.strictEqual(n.pa.pending, null); assert.strictEqual(n.pa.waiters.length, 0); assert.strictEqual(longTimers(1000), t0long, "request timer was leaked");
  });
  await test("channel closes while A is still waiting for B's HELLO (this took the full timeout before Stage 6)", async () => {
    const n = nodes({ timeoutMs: 5000 });                                              // no HELLO exchanged: A sits in waitReady()
    const p = n.pa.infer(C[0].ids).catch((e) => e); await sleep(50); const t = Date.now();
    n.pa.dispose("DataChannel closed"); const e = await p;
    assert.strictEqual(e.code, "CHANNEL_CLOSED"); assert(Date.now() - t < 300, "took " + (Date.now() - t) + " ms"); assert.strictEqual(n.pa.waiters.length, 0);
  });
  await test("channel dies in the MIDDLE of a chunked transmission: send error surfaces, B's half-message later expires", async () => {
    const n = await connected({ max: 16384, reassemblyTimeoutMs: 150, timeoutMs: 5000 });
    n.link.failAfter = n.link.sentCount + 2;                                             // two frames get through, the third send throws
    await rejects(n.pa.infer(C[4].ids), "CHANNEL_CLOSED"); assert.strictEqual(n.pa.pending, null);
    await until(() => n.pb.reassembler.inProgress); await until(() => !n.pb.reassembler.inProgress, 1500);   // B held a partial message, then discarded it
    assert.strictEqual(n.pb.stats.expired, 1);
  });

  section("[Node simulation] 2. Inference request times out");
  await test("B silent: TIMEOUT, pending slot and timers released, next request is clean", async () => {
    const t0long = longTimers(250); const n = await connected({ timeoutMs: 300 }); let mute = true;
    n.link.filter = (dir, buf) => (dir === "AtoB" && mute ? null : buf);
    await rejects(n.pa.infer(C[0].ids), "TIMEOUT"); assert.strictEqual(n.pa.pending, null); assert.strictEqual(longTimers(250), t0long, "timer leaked after timeout");
    mute = false; const r = await n.pa.infer(C[0].ids); assert.strictEqual(r.token, C[0].argmax);
  });

  section("[Node simulation] 3. Invalid activation packet is received");
  for (const [name, code, mutate] of [
    ["NaN inside the activation", "NON_FINITE", (p) => P.encodeActivation({ sessionId: BigInt(p.sessionId), seq: p.seq, activations: Float32Array.from({ length: 2 * D }, () => NaN), shape: [1, 2, D] })],
    ["wrong tensor rank [T,384]", "BAD_SHAPE", (p) => P.encodeActivation({ sessionId: BigInt(p.sessionId), seq: p.seq, activations: new Float32Array(2 * D), shape: [2, D] })],
    ["one flipped payload bit (checksum)", "CHECKSUM", (p, buf) => { new Uint8Array(buf)[200] ^= 8; return buf; }],
  ]) {
    await test(`${name}: B answers with ERROR ${code}, A fails fast with that code, B computes nothing, next request is fine`, async () => {
      const sB = spy(realB), n = await connected({ engineB: sB });
      n.link.filter = (dir, buf) => { if (dir !== "AtoB") return buf; const p = P.decodePacket(buf.slice(0)); return p.msgType === P.MSG_TYPES.ACTIVATION ? mutate(p, buf) : buf; };
      const t = Date.now(); await rejects(n.pa.infer(C[1].ids), code, { fromPeer: true }); assert(Date.now() - t < 1500); assert.strictEqual(sB.calls, 0);
      n.link.filter = null; assert.strictEqual((await n.pa.infer(C[1].ids)).token, C[1].argmax); assert.strictEqual(sB.calls, 1);
    });
  }

  section("[Node simulation] 4. Duplicate response arrives");
  await test("a duplicated reply resolves the request once; a stray copy landing during the NEXT request cannot hijack it", async () => {
    const n = await connected(); let first = true;
    n.link.filter = (dir, buf) => { if (dir === "BtoA" && first && msgType(buf.slice(0)) === P.MSG_TYPES.TOKEN) { first = false; const copy = buf.slice(0); setTimeout(() => n.link.toA(copy), 8); } return buf; };
    const r1 = await n.pa.infer(C[2].ids); assert.strictEqual(r1.token, C[2].argmax);
    const r2 = await n.pa.infer(C[5].ids);                                               // T=200: slow enough that the stray copy arrives while this is pending
    assert.strictEqual(r2.token, C[5].argmax, "the duplicate of request 1 changed request 2's result");
    assert(n.evA.filter((e) => e.t === "warn" && /ignored a reply/.test(e.message)).length >= 1, "the duplicate should have been ignored with a warning");
    assert.strictEqual(n.pa.pending, null);
  });

  section("[Node simulation] 5. Late response arrives");
  await test("reply arrives after A gave up; it is ignored, and the NEXT request gets its own answer", async () => {
    const n = await connected({ timeoutMs: 150 }); let delayed = false;
    n.link.filter = (dir, buf) => { if (dir === "BtoA" && !delayed && msgType(buf.slice(0)) === P.MSG_TYPES.TOKEN) { delayed = true; const c = buf.slice(0); setTimeout(() => n.link.toA(c), 400); return null; } return buf; };
    await rejects(n.pa.infer(C[0].ids), "TIMEOUT");
    const p2 = n.pa.infer(C[3].ids, { timeoutMs: 5000 });                                // request 2 is pending when request 1's late reply lands
    const r2 = await p2; assert.strictEqual(r2.token, C[3].argmax);
    await sleep(450);                                                                    // the late reply (token for C[0]) has now arrived
    assert(n.evA.some((e) => e.t === "warn" && /ignored a reply/.test(e.message)), "late reply should be ignored with a warning");
    assert.strictEqual(n.pa.pending, null); assert.strictEqual((await n.pa.infer(C[3].ids)).token, C[3].argmax);
  });

  section("[Node simulation] 6. Incomplete chunk transfer expires");
  await test("half a message, then silence: discarded after the timeout, warning raised, next message processed normally", async () => {
    const n = await connected({ max: 16384, reassemblyTimeoutMs: 120 });
    const act = realA.forward(C[4].ids), pkt = P.encodeActivation({ sessionId: 500n, seq: 1, activations: act, shape: [1, C[4].T, D] });
    const frames = P.splitIntoChunks(pkt, 16384, 77); assert(frames.length >= 3);
    await n.pb.handleMessage(frames[0]); await n.pb.handleMessage(frames[1]); assert.strictEqual(n.pb.reassembler.inProgress, true);
    await sleep(60); await n.pb.handleMessage(frames[2] || frames[1]);                   // activity re-arms the (inactivity) timer; it must NOT expire yet
    assert.strictEqual(n.pb.stats.expired, 0, "expired too early");
    await until(() => n.pb.stats.expired === 1, 1500);
    assert.strictEqual(n.pb.reassembler.inProgress, false); assert(n.evB.some((e) => e.t === "warn" && /incomplete chunked message/.test(e.message)));
    assert.strictEqual((await n.pa.infer(C[4].ids)).token, C[4].argmax);                 // the pipeline is not wedged
  });
  await test("a COMPLETE transfer leaves no expiry timer behind", async () => {
    const n = await connected({ max: 16384, reassemblyTimeoutMs: 700, timeoutMs: 6000 });
    const base = longTimers(500); await n.pa.infer(C[4].ids); assert.strictEqual(longTimers(500), base, "a reassembly/request timer survived a finished request");
  });
  await test("a broken chunk produces ONE error, not one per remaining frame, and cannot fail the next request", async () => {
    const n = await connected({ max: 16384 }); const errors = () => n.link.log.BtoA.filter((b) => msgType(b.slice(0)) === P.MSG_TYPES.ERROR).length;
    const pkt = P.encodeActivation({ sessionId: 600n, seq: 1, activations: realA.forward(C[5].ids), shape: [1, C[5].T, D] });
    const frames = P.splitIntoChunks(pkt, 16384, 88); assert(frames.length >= 5);
    await n.pb.handleMessage(frames[0]); for (let i = 2; i < frames.length; i++) await n.pb.handleMessage(frames[i]);   // frame 1 "lost in transit"
    await sleep(30); assert.strictEqual(errors(), 1, "expected exactly one ERROR packet, got " + errors());
    assert.strictEqual((await n.pa.infer(C[1].ids)).token, C[1].argmax, "the leftovers of the broken transfer hurt the next request");
  });

  section("[Node simulation] 7. Peer disconnects during inference");
  await test("B disappears mid-request: B's late reply fails cleanly (no unhandled rejection), A is released when it learns the channel closed", async () => {
    const n = await connected({ engineB: spy(realB, () => n_pb_dispose()), timeoutMs: 5000 });
    function n_pb_dispose() { n.pb.dispose("peer went away"); }                          // B's page is closed while its compute is running
    const p = n.pa.infer(C[1].ids).catch((e) => e); await until(() => n.pb.disposed); await sleep(30);
    assert.strictEqual(n.link.log.BtoA.length, 1, "B must send nothing after it is disposed (only its HELLO)");
    n.pa.dispose("DataChannel closed");                                                  // A's browser reports the closure
    const e = await p; assert.strictEqual(e.code, "CHANNEL_CLOSED"); assert.strictEqual(n.pa.pending, null);
  });

  section("[Node simulation] 8. Model fingerprint mismatch");
  await test("A refuses at once with MODEL_MISMATCH (no 20 s wait), sends no activation, B computes nothing, nothing leaks", async () => {
    const sB = spy(realB), t0long = longTimers(1000), n = await connected({ fpB: FP.replace(/b\d+$/, "b1"), engineB: sB, timeoutMs: 5000 });
    const t = Date.now(); await rejects(n.pa.infer(C[0].ids), "MODEL_MISMATCH"); assert(Date.now() - t < 500);
    assert.strictEqual(sB.calls, 0); assert.strictEqual(n.link.log.AtoB.length, 1, "only HELLO may have been sent");
    assert.strictEqual(n.pa.pending, null); assert.strictEqual(n.pa.waiters.length, 0); assert.strictEqual(longTimers(1000), t0long);
  });

  section("[Node simulation] 9. Reconnection creates a valid new session");
  await test("old pipelines are dead; a new pair works, uses new session ids, and an old session's reply cannot touch it", async () => {
    const sB1 = spy(realB), n1 = await connected({ engineB: sB1 }); const r1 = await n1.pa.infer(C[0].ids, { wantLogits: true });
    n1.link.close(); n1.pa.dispose("DataChannel closed"); n1.pb.dispose("DataChannel closed");                       // what main.js does on link loss
    await rejects(n1.pa.infer(C[0].ids), "CHANNEL_CLOSED"); await rejects(n1.pa.sendHello(), "CHANNEL_CLOSED");   // a dead pipeline cannot send...
    const calls = sB1.calls; await n1.pb.handleMessage(new Uint8Array(P.encodeActivation({ sessionId: 1n, seq: 1, activations: new Float32Array(D), shape: [1, 1, D] })).buffer);
    assert.strictEqual(sB1.calls, calls, "...and a dead B must not compute");
    const n2 = await connected(); const p2 = n2.pa.infer(C[5].ids, { wantLogits: true });                           // the NEW session
    await n2.pa.handleMessage(P.encodeToken({ sessionId: BigInt(r1.sessionId), seq: 1, token: 7 }));                   // a straggler from the OLD session
    const r2 = await p2; assert.notStrictEqual(r2.sessionId, r1.sessionId); assert.strictEqual(r2.token, C[5].argmax);
    const want = local.partitionB.forward(local.partitionA.forward(C[5].ids), C[5].T); assert(valuesEqual(r2.logits, want.logits), "new session result differs from local inference");
    assert(n2.evA.some((e) => e.t === "warn" && /ignored a reply/.test(e.message)));
  });

  section("[Node simulation] 10. Cleanup is called multiple times");
  await test("dispose() x3 while a request is in flight: one rejection, one 'disposed' event, no timers, no waiters, later calls are no-ops", async () => {
    const base = longTimers(1000), n = await connected({ timeoutMs: 5000 }); n.link.filter = (dir, buf) => (dir === "AtoB" ? null : buf);
    let rejections = 0; const p = n.pa.infer(C[1].ids).catch((e) => { rejections++; return e; }); await sleep(50);
    assert.strictEqual(n.pa.dispose("first"), true); assert.strictEqual(n.pa.dispose("second"), false); assert.strictEqual(n.pa.dispose("third"), false);
    const e = await p; assert.strictEqual(rejections, 1); assert.strictEqual(e.code, "CHANNEL_CLOSED"); assert.strictEqual(e.message, "first", "the FIRST reason wins");
    assert.strictEqual(n.evA.filter((x) => x.t === "disposed").length, 1); assert.strictEqual(n.pa.pending, null); assert.strictEqual(n.pa.waiters.length, 0);
    assert.strictEqual(longTimers(1000), base, "a timer survived dispose()");
    const events = n.evA.length; await n.pa.handleMessage(P.encodeToken({ sessionId: 1n, seq: 1, token: 5 })); assert.strictEqual(n.evA.length, events, "a disposed pipeline must ignore messages");
    await rejects(n.pa.infer(C[1].ids), "CHANNEL_CLOSED");
    n.pb.dispose("x"); n.pb.dispose("y"); assert.strictEqual(n.pb.seen.size, 0);
  });
  await test("dispose() on a pipeline that never did anything, and in the middle of an unfinished chunk transfer", async () => {
    const n = nodes(); assert.strictEqual(n.pb.dispose("idle"), true); assert.strictEqual(n.pb.dispose("idle again"), false);
    const m = await connected({ max: 16384, reassemblyTimeoutMs: 5000 }); const base = longTimers(1000);
    const frames = P.splitIntoChunks(P.encodeActivation({ sessionId: 9n, seq: 1, activations: realA.forward(C[4].ids), shape: [1, C[4].T, D] }), 16384, 5);
    await m.pb.handleMessage(frames[0]); assert.strictEqual(m.pb.reassembler.inProgress, true);
    m.pb.dispose("closed"); assert.strictEqual(m.pb.reassembler.inProgress, false); assert(longTimers(1000) <= base);
  });

  console.log("\n======================================================================");
  console.log(failed ? `STAGE 6 TESTS FAILED: ${passed} passed, ${failed} failed` : `STAGE 6 TESTS PASSED: ${passed} passed, 0 failed`);
  console.log("These are unit tests and Node simulations. They are NOT browser, two-tab or physical-device evidence.");
  console.log("======================================================================");
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
