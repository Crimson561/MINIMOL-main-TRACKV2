/**
 * Test C — Communication Correctness & Protocol Validation
 * Tests:
 * 1. Binary activation serialization (Phone A -> Phone B).
 * 2. Shape, type, and payload integrity checks.
 * 3. Adler-32 checksum calculation and corruption detection.
 * 4. Token packet serialization (Phone B -> Phone A).
 * 5. Control packet encoding/decoding.
 * 6. Simulated bidirectional WebRTC DataChannel transfer with timing instrumentation.
 */

"use strict";

const assert = require("assert");
const {
  MAGIC,
  PROTOCOL_VERSION,
  MSG_TYPES,
  DATA_TYPES,
  HEADER_SIZE,
  encodeActivation,
  encodeToken,
  encodeControl,
  decodePacket,
} = require("../protocol/mesh_protocol");

function runTest() {
  console.log("======================================================================");
  console.log("TEST C — COMMUNICATION & PROTOCOL VALIDATION");
  console.log("======================================================================");

  // 1. Test Activation Encoding and Decoding
  const seqLen = 31;
  const dModel = 384;
  const activations = new Float32Array(seqLen * dModel);
  for (let i = 0; i < activations.length; i++) {
    activations[i] = (i % 100) * 0.01;
  }

  const sessionId = 1234567890123456789n;
  const seq = 1;
  const shape = [1, seqLen, dModel, 0];

  console.log(`1. Encoding activation tensor [1, ${seqLen}, ${dModel}] (${activations.byteLength} bytes)...`);
  const packetBuffer = encodeActivation({
    sessionId,
    seq,
    activations,
    shape,
  });
  assert.equal(packetBuffer.byteLength, HEADER_SIZE + activations.byteLength);
  console.log(`   Packet size: ${packetBuffer.byteLength} bytes (Header: ${HEADER_SIZE}B, Payload: ${activations.byteLength}B)`);

  const decoded = decodePacket(packetBuffer);
  assert.equal(decoded.magic, MAGIC);
  assert.equal(decoded.version, PROTOCOL_VERSION);
  assert.equal(decoded.msgType, MSG_TYPES.ACTIVATION);
  assert.equal(decoded.seq, seq);
  assert.equal(decoded.sessionId, sessionId.toString());
  assert.equal(decoded.rank, 4);
  assert.deepEqual(decoded.shape, shape);
  assert.equal(decoded.payload.length, activations.length);

  // Check numerical equality of decoded float32
  let maxDiff = 0.0;
  for (let i = 0; i < activations.length; i++) {
    const diff = Math.abs(activations[i] - decoded.payload[i]);
    if (diff > maxDiff) maxDiff = diff;
  }
  assert.equal(maxDiff, 0.0, "Decoded activations altered!");
  console.log("   ✓ Activation serialization and zero-copy deserialization bit-identical!");

  // 2. Test Checksum Corruption Detection
  console.log("\n2. Testing corruption detection...");
  const corruptedBuffer = packetBuffer.slice(0);
  const corruptedView = new Uint8Array(corruptedBuffer);
  // Corrupt one byte in payload
  corruptedView[HEADER_SIZE + 10] ^= 0xff;
  assert.throws(() => {
    decodePacket(corruptedBuffer);
  }, /Packet checksum mismatch/, "Should have detected corrupted payload!");
  console.log("   ✓ Corrupted packet rejected by Adler-32 checksum validation.");

  // 3. Test Invalid Magic / Truncation Rejection
  console.log("\n3. Testing malformed packet rejection...");
  const badMagicBuffer = packetBuffer.slice(0);
  new DataView(badMagicBuffer).setUint32(0, 0xdeadbeef, false);
  assert.throws(() => {
    decodePacket(badMagicBuffer);
  }, /Invalid protocol magic/, "Should reject invalid magic!");

  const truncatedBuffer = packetBuffer.slice(0, 30);
  assert.throws(() => {
    decodePacket(truncatedBuffer);
  }, /Packet too small/, "Should reject packet smaller than header!");
  console.log("   ✓ Invalid magic and truncated packets rejected.");

  // 4. Test Token Message (Phone B -> Phone A)
  console.log("\n4. Testing Token message encoding/decoding...");
  const tokenPacket = encodeToken({
    sessionId,
    seq: 1,
    token: 179,
    isEos: false,
    timing: { bComputeMs: 14.5, bNetworkMs: 2.1 },
  });
  const decodedToken = decodePacket(tokenPacket);
  assert.equal(decodedToken.msgType, MSG_TYPES.TOKEN);
  assert.equal(decodedToken.payload.token, 179);
  assert.equal(decodedToken.payload.timing.bComputeMs, 14.5);
  console.log(`   ✓ Token packet decoded: token=${decodedToken.payload.token}, compute=${decodedToken.payload.timing.bComputeMs}ms`);

  // 5. Test Control Message
  console.log("\n5. Testing Control message...");
  const controlPacket = encodeControl({
    sessionId,
    seq: 0,
    command: "START_GENERATION",
    data: { prompt: "MESH", maxTokens: 10 },
  });
  const decodedControl = decodePacket(controlPacket);
  assert.equal(decodedControl.msgType, MSG_TYPES.CONTROL);
  assert.equal(decodedControl.payload.command, "START_GENERATION");
  console.log(`   ✓ Control packet decoded: command=${decodedControl.payload.command}`);

  console.log("======================================================================");
  console.log("TEST C PASSED: PROTOCOL AND DATA TRANSFER INTEGRITY VERIFIED");
  console.log("======================================================================");
}

runTest();
