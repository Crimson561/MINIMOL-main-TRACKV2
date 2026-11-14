/**
 * Stage 4 — Tensor serialization & reconstruction tests (pure Node, no WebRTC).
 * Run:  node tests/test_tensor.js
 *
 * Part 1: encodeTensor / decodeTensor on their own.
 * Part 2: a simulated A <-> B channel (bytes are copied into a fresh ArrayBuffer, as the
 *         network does). The real DataChannel test is `meshTest.tensor()` in the browser.
 */
"use strict";

const assert = require("assert");
const P = require("../protocol/mesh_protocol");
const { MSG_TYPES, DATA_TYPES, HEADER_SIZE, adler32, encodeTensor, decodeTensor, encodeActivation, decodePacket } = P;

let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log("  ✓ " + name); }
  catch (e) { failed++; console.log("  ✗ " + name + "\n      " + e.message); }
}

// ---------- helpers ----------
const range = (n, f) => Float32Array.from({ length: n }, (_, i) => f(i));
const bits = (f32) => new Uint32Array(f32.buffer, f32.byteOffset, f32.length); // raw bit patterns
function assertBitIdentical(a, b, msg = "") {
  assert.strictEqual(a.length, b.length, msg + " length differs");
  const x = bits(a), y = bits(b);
  for (let i = 0; i < x.length; i++) {
    if (x[i] !== y[i]) assert.fail(`${msg} element ${i}: 0x${x[i].toString(16)} != 0x${y[i].toString(16)}`);
  }
}
const networkCopy = (ab) => ab.slice(0); // what a transport delivers: a brand-new ArrayBuffer
function roundTrip(data, shape, opts) {
  const packet = encodeTensor({ sessionId: 42n, seq: 7, data, shape });
  const t = decodeTensor(networkCopy(packet), opts);
  assert.deepStrictEqual(t.shape, shape);
  assert.strictEqual(t.dtype, "float32");
  assert.strictEqual(t.numel, data.length);
  assertBitIdentical(t.data, data);
  return t;
}
// Build a packet, apply a header edit, and (optionally) fix the payload checksum.
function forge(data, shape, edit, recompute = false) {
  const b = encodeTensor({ data, shape });
  edit(new DataView(b), new Uint8Array(b));
  if (recompute) {
    const v = new DataView(b);
    v.setUint32(44, adler32(new Uint8Array(b), HEADER_SIZE, v.getUint32(40, false)), false);
  }
  return b;
}

console.log("\nPART 1 — serialization and reconstruction (no WebRTC)");

test("1-D tensor", () => roundTrip(range(8, (i) => i * 0.25), [8]));
test("2-D tensor", () => roundTrip(range(12, (i) => i - 5.5), [3, 4]));
test("3-D tensor (batch, time, channels)", () => roundTrip(range(24, (i) => Math.fround(Math.sin(i))), [2, 3, 4]));
test("4-D tensor (uses all four shape slots)", () => roundTrip(range(30, (i) => i), [1, 2, 3, 5]));
test("all zeros", () => roundTrip(new Float32Array(10), [2, 5]));
test("negative and positive values", () => roundTrip(Float32Array.of(-3.5, 2, -0.001, 1e6, -1e-6, 0.75), [2, 3]));
test("special values survive bit-for-bit: NaN, ±Infinity, -0, denormal, FLT_MAX", () => {
  const d = Float32Array.of(NaN, Infinity, -Infinity, -0, 1e-45, 3.4028234663852886e38);
  const t = roundTrip(d, [6]);
  assert.ok(Object.is(t.data[3], -0), "-0 became +0");
});
test("same size, different shapes stay distinguishable ([6] vs [2,3] vs [3,2])", () => {
  const d = range(6, (i) => i);
  for (const s of [[6], [2, 3], [3, 2], [1, 6], [6, 1]]) assert.deepStrictEqual(roundTrip(d, s).shape, s);
});
test("scalar: shape [] is rank 0 with 1 element", () => roundTrip(Float32Array.of(3.25), []));
test("empty tensors: [0], [0,5], [2,0,3] (rank is preserved, no payload)", () => {
  for (const s of [[0], [0, 5], [2, 0, 3]]) {
    const packet = encodeTensor({ data: new Float32Array(0), shape: s });
    assert.strictEqual(packet.byteLength, HEADER_SIZE);
    const t = decodeTensor(packet);
    assert.deepStrictEqual(t.shape, s);
    assert.strictEqual(t.data.length, 0);
  }
});
test("float64 source rounded to float32: compare with a tolerance, not equality", () => {
  const src = [Math.PI, -Math.E, 1 / 3, 123456.789];
  const f32 = Float32Array.from(src); // rounding happens HERE, before any serialization
  const t = roundTrip(f32, [4]);
  src.forEach((v, i) => assert.ok(Math.abs(t.data[i] - v) <= Math.abs(v) * 1.2e-7 + 1e-30, `element ${i}`));
});
test("large tensor [1, 31, 384] (the future activation size)", () => roundTrip(range(31 * 384, (i) => (i % 97) - 40.5), [1, 31, 384]));
test("encoding does not modify the input", () => {
  const d = range(6, (i) => i), copy = d.slice();
  encodeTensor({ data: d, shape: [6] });
  assertBitIdentical(d, copy);
});
test("a subarray (non-zero byteOffset) encodes only its own elements", () => {
  const base = range(10, (i) => i), view = base.subarray(2, 6);
  const t = roundTrip(view, [4]);
  assert.deepStrictEqual(Array.from(t.data), [2, 3, 4, 5]);
});
test("metadata: sessionId, seq, TENSOR message type, FP32 dtype code", () => {
  const b = encodeTensor({ sessionId: 99n, seq: 5, data: Float32Array.of(1), shape: [1] });
  const p = decodePacket(b);
  assert.strictEqual(p.msgType, MSG_TYPES.TENSOR);
  assert.strictEqual(p.dataType, DATA_TYPES.FP32);
  assert.strictEqual(p.seq, 5);
  assert.strictEqual(p.sessionId, "99");
});
test("byte layout: header big-endian, payload little-endian (1.0f -> 00 00 80 3F)", () => {
  const u8 = new Uint8Array(encodeTensor({ data: Float32Array.of(1), shape: [1] }));
  assert.deepStrictEqual(Array.from(u8.subarray(0, 4)), [0x4d, 0x45, 0x53, 0x48]); // "MESH"
  assert.deepStrictEqual(Array.from(u8.subarray(24, 28)), [0, 0, 0, 1]);          // dim0 = 1, big-endian
  assert.deepStrictEqual(Array.from(u8.subarray(HEADER_SIZE)), [0x00, 0x00, 0x80, 0x3f]);
});

console.log("\n  Alignment and copying");
test("aligned packet in a larger buffer: zero-copy view (shares memory)", () => {
  const pkt = new Uint8Array(encodeTensor({ data: range(4, (i) => i), shape: [4] }));
  const big = new Uint8Array(pkt.length + 8); big.set(pkt, 4); // starts at offset 4 (aligned)
  const t = decodeTensor(big.subarray(4, 4 + pkt.length));
  assert.strictEqual(t.data.buffer, big.buffer);
  assert.deepStrictEqual(Array.from(t.data), [0, 1, 2, 3]);
});
test("UNALIGNED packet (offset 1): falls back to a copy instead of throwing", () => {
  const pkt = new Uint8Array(encodeTensor({ data: range(4, (i) => i + 0.5), shape: [4] }));
  const big = new Uint8Array(pkt.length + 2); big.set(pkt, 1);
  const t = decodeTensor(big.subarray(1, 1 + pkt.length));
  assert.notStrictEqual(t.data.buffer, big.buffer);
  assert.deepStrictEqual(Array.from(t.data), [0.5, 1.5, 2.5, 3.5]);
});
test("copy:true gives independent memory", () => {
  const pkt = encodeTensor({ data: range(4, (i) => i), shape: [4] });
  const t = decodeTensor(pkt, { copy: true });
  t.data[0] = 999;
  assert.strictEqual(new Float32Array(pkt, HEADER_SIZE, 4)[0], 0);
});

console.log("\n  Invalid input to encodeTensor");
const F = Float32Array.of(1, 2, 3, 4, 5, 6);
test("unsupported types rejected: Float64Array, Int32Array, Uint8Array, Array, null", () => {
  for (const bad of [new Float64Array(6), new Int32Array(6), new Uint8Array(6), [1, 2, 3, 4, 5, 6], null, undefined]) {
    assert.throws(() => encodeTensor({ data: bad, shape: [6] }), /Unsupported tensor type/);
  }
});
test("shape product must equal data length", () => {
  assert.throws(() => encodeTensor({ data: F, shape: [2, 2] }), /holds 4 elements but data has 6/);
  assert.throws(() => encodeTensor({ data: F, shape: [] }), /holds 1 elements/);
});
test("rank above 4 is rejected (it used to silently corrupt the header)", () => {
  assert.throws(() => encodeTensor({ data: new Float32Array(32), shape: [2, 2, 2, 2, 2] }), /rank 5 exceeds/);
});
test("negative, fractional, NaN and huge dimensions rejected", () => {
  for (const s of [[-1, -6], [2.5, 2.4], [NaN], [2 ** 32], ["6"]]) {
    assert.throws(() => encodeTensor({ data: F, shape: s }), /Invalid dimension/);
  }
});
test("shape must be an array; seq must be a uint32", () => {
  assert.throws(() => encodeTensor({ data: F, shape: 6 }), /shape must be an array/);
  assert.throws(() => encodeTensor({ data: F, shape: [6], seq: -1 }), /Invalid seq/);
  assert.throws(() => encodeTensor({ data: F, shape: [6], seq: 1.5 }), /Invalid seq/);
});

console.log("\n  Invalid packets given to decodeTensor");
test("shape edited in the header: payload length no longer matches", () => {
  const b = forge(F, [2, 3], (v) => v.setUint32(28, 4, false)); // [2,3] -> [2,4]; checksum (payload only) still valid
  assert.throws(() => decodeTensor(b), /Payload length mismatch: shape \[2, 4\] of float32 needs 32 bytes, header says 24/);
});
test("payload length edited to a non-multiple of 4", () => {
  const b = forge(F, [6], (v) => v.setUint32(40, 10, false), true);
  assert.throws(() => decodeTensor(b), /Payload length mismatch/);
});
test("rank forged to 5 (reads past the shape slots)", () => {
  assert.throws(() => decodeTensor(forge(F, [6], (v) => v.setUint8(21, 5))), /Invalid tensor rank 5/);
});
test("truncated packet (payload cut short)", () => {
  assert.throws(() => decodeTensor(encodeTensor({ data: F, shape: [6] }).slice(0, HEADER_SIZE + 10)), /Truncated packet/);
});
test("header-only fragment", () => {
  assert.throws(() => decodeTensor(encodeTensor({ data: F, shape: [6] }).slice(0, 20)), /Packet too small/);
});
test("extra trailing bytes", () => {
  const pkt = new Uint8Array(encodeTensor({ data: F, shape: [6] }));
  const longer = new Uint8Array(pkt.length + 4); longer.set(pkt);
  assert.throws(() => decodeTensor(longer), /Packet size mismatch/);
});
test("corrupted payload byte (checksum)", () => {
  const b = encodeTensor({ data: F, shape: [6] });
  new Uint8Array(b)[HEADER_SIZE + 3] ^= 0x40;
  assert.throws(() => decodeTensor(b), /checksum mismatch/);
});
test("unsupported dtype codes: UINT8, INT32, UTF8_JSON, unknown", () => {
  for (const code of [DATA_TYPES.UINT8, DATA_TYPES.INT32, DATA_TYPES.UTF8_JSON, 0x7f]) {
    assert.throws(() => decodeTensor(forge(F, [6], (v) => v.setUint8(20, code))), /Unsupported tensor dtype code/);
  }
});
test("a non-tensor packet (ACTIVATION) is refused", () => {
  const act = encodeActivation({ seq: 0, activations: F, shape: [1, 6] });
  assert.throws(() => decodeTensor(act), /Not a tensor packet: message type 0x01/);
});
test("bad magic / bad version / wrong input type", () => {
  assert.throws(() => decodeTensor(forge(F, [6], (v) => v.setUint32(0, 0xdeadbeef))), /Invalid protocol magic/);
  assert.throws(() => decodeTensor(forge(F, [6], (v) => v.setUint8(4, 9))), /Unsupported protocol version/);
  assert.throws(() => decodeTensor("hello"), /Invalid raw packet type/);
});

console.log("\n  Backward compatibility with the existing protocol");
test("encodeActivation / decodePacket behave as before (ACTIVATION still yields a Float32Array)", () => {
  const d = range(8, (i) => i);
  const p = decodePacket(encodeActivation({ sessionId: 1n, seq: 3, activations: d, shape: [1, 2, 4] }));
  assert.strictEqual(p.msgType, MSG_TYPES.ACTIVATION);
  assert.ok(p.payload instanceof Float32Array);
  assertBitIdentical(p.payload, d);
});
test("decodePacket also accepts a TENSOR packet (payload as raw bytes)", () => {
  const p = decodePacket(encodeTensor({ data: F, shape: [6] }));
  assert.strictEqual(p.msgType, MSG_TYPES.TENSOR);
  assert.strictEqual(p.payload.byteLength, 24);
});

console.log("\nPART 2 — simulated A <-> B transfer (stand-in for the DataChannel)");
// Each "device" only sees bytes delivered through channel(); no shared objects.
function channel(bytes) { return bytes.slice(0); }
function deviceB_receiveAndEcho(bytes) {
  const t = decodeTensor(bytes);                                   // reconstruct
  return encodeTensor({ sessionId: BigInt(t.sessionId), seq: t.seq, flags: 2, data: t.data, shape: t.shape }); // echo
}
const cases = [
  ["1-D", range(8, (i) => i - 3.5), [8]],
  ["2-D", range(12, (i) => i * -0.5), [3, 4]],
  ["3-D", range(24, (i) => Math.fround(Math.cos(i))), [2, 3, 4]],
  ["zeros", new Float32Array(6), [2, 3]],
  ["empty", new Float32Array(0), [0]],
];
for (const [from, to] of [["A", "B"], ["B", "A"]]) {
  test(`${from} -> ${to} -> ${from}: every case returns bit-identical with the same shape`, () => {
    cases.forEach(([name, data, shape], idx) => {
      const sent = encodeTensor({ sessionId: 1n, seq: idx, data, shape });
      const echoed = decodeTensor(channel(deviceB_receiveAndEcho(channel(sent))));
      assert.deepStrictEqual(echoed.shape, shape, name);
      assert.strictEqual(echoed.seq, idx, name);
      assert.strictEqual(echoed.flags, 2, name);
      assertBitIdentical(echoed.data, data, name);
    });
  });
}

console.log(`\n${failed === 0 ? "TENSOR TESTS PASSED" : "TENSOR TESTS FAILED"}: ${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
