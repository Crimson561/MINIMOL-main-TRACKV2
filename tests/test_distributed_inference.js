/**
 * Test D — Distributed Inference Across Two Independent Nodes
 * Simulates Phone A (Partition A) and Phone B (Partition B) communicating
 * solely through binary DataChannel messages.
 * Compares final tokens against PyTorch reference model.
 */

"use strict";

const fs = require("fs");
const path = require("path");
const assert = require("assert");

const { MeshPartitionA, MeshPartitionB } = require("../engine/inference_engine");
const {
  MSG_TYPES,
  encodeActivation,
  encodeToken,
  decodePacket,
} = require("../protocol/mesh_protocol");

const WEIGHTS_DIR = path.join(__dirname, "..", "room-server", "public", "weights");

// Mock WebRTC DataChannel connection between two endpoints
class MockDataChannel {
  constructor(name) {
    this.name = name;
    this.peer = null;
    this.onmessage = null;
    this.bytesSent = 0;
    this.latencyMs = 2.0; // Simulated network delay
  }

  connect(peerChannel) {
    this.peer = peerChannel;
    peerChannel.peer = this;
  }

  send(data) {
    if (!this.peer) throw new Error("DataChannel not connected");
    const byteLen = data.byteLength || data.length;
    this.bytesSent += byteLen;
    // Deliver asynchronously to simulate real network
    setTimeout(() => {
      if (this.peer.onmessage) {
        this.peer.onmessage({ data });
      }
    }, this.latencyMs);
  }
}

async function runDistributedInferenceTest() {
  console.log("======================================================================");
  console.log("TEST D — DISTRIBUTED INFERENCE TEST (PHONE A + PHONE B)");
  console.log("======================================================================");

  // 1. Load weights
  const manifest = JSON.parse(fs.readFileSync(path.join(WEIGHTS_DIR, "weights_manifest.json"), "utf-8"));
  const partABuf = fs.readFileSync(path.join(WEIGHTS_DIR, "partition_a.bin"));
  const partBBuf = fs.readFileSync(path.join(WEIGHTS_DIR, "partition_b.bin"));

  const abA = partABuf.buffer.slice(partABuf.byteOffset, partABuf.byteOffset + partABuf.byteLength);
  const abB = partBBuf.buffer.slice(partBBuf.byteOffset, partBBuf.byteOffset + partBBuf.byteLength);

  // Phone A: ONLY loads Partition A
  console.log("Initializing Phone A (Layers 0..2)...");
  const phoneA = {
    engine: MeshPartitionA.fromBuffer(abA, manifest),
    context: Array.from(Buffer.from("MESH", "utf-8")),
    channel: new MockDataChannel("PhoneA_DC"),
    seq: 0,
    sessionId: 9876543210n,
    metrics: {
      aComputeTimes: [],
      serializationTimes: [],
      networkTimes: [],
      tokenLatencies: [],
      bytesTransferred: 0,
    },
  };

  // Phone B: ONLY loads Partition B
  console.log("Initializing Phone B (Layers 3..5 + Head)...");
  const phoneB = {
    engine: MeshPartitionB.fromBuffer(abB, manifest),
    channel: new MockDataChannel("PhoneB_DC"),
    metrics: {
      bComputeTimes: [],
    },
  };

  // Establish direct DataChannel link
  phoneA.channel.connect(phoneB.channel);
  console.log("P2P DataChannel established between Phone A and Phone B.\n");

  const expectedSequence = [77, 69, 83, 72, 228, 228, 228, 228, 228, 228, 98, 98, 240, 98];
  const stepsToGenerate = 10;

  // Set up Phone B receiver
  phoneB.channel.onmessage = (event) => {
    const recvTime = performance.now();
    const packet = decodePacket(event.data);
    assert.equal(packet.msgType, MSG_TYPES.ACTIVATION, "Phone B expects activation packet");

    // Phone B executes layers 3..5 + lm_head
    const bStart = performance.now();
    const T = packet.shape[1];
    const { nextByte, logits } = phoneB.engine.forward(packet.payload, T);
    const bEnd = performance.now();
    const bComputeTime = bEnd - bStart;
    phoneB.metrics.bComputeTimes.push(bComputeTime);

    // Phone B encodes next token and sends back directly to Phone A
    const tokenPacket = encodeToken({
      sessionId: packet.sessionId,
      seq: packet.seq,
      token: nextByte,
      isEos: false,
      timing: { bComputeTime },
    });
    phoneB.channel.send(tokenPacket);
  };

  // Run distributed generation loop
  const totalGenStart = performance.now();

  for (let step = 0; step < stepsToGenerate; step++) {
    const stepStart = performance.now();
    phoneA.seq++;

    // 1. Phone A executes layers 0..2
    const aComputeStart = performance.now();
    const activations = phoneA.engine.forward(phoneA.context);
    const aComputeEnd = performance.now();
    const aComputeTime = aComputeEnd - aComputeStart;
    phoneA.metrics.aComputeTimes.push(aComputeTime);

    // 2. Phone A serializes activation tensor
    const serStart = performance.now();
    const activationPacket = encodeActivation({
      sessionId: phoneA.sessionId,
      seq: phoneA.seq,
      activations,
      shape: [1, phoneA.context.length, 384, 0],
    });
    const serEnd = performance.now();
    phoneA.metrics.serializationTimes.push(serEnd - serStart);
    phoneA.metrics.bytesTransferred += activationPacket.byteLength;

    // 3. Phone A sends packet through direct DataChannel and awaits Phone B token response
    const receivedToken = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("DataChannel response timeout")), 5000);
      phoneA.channel.onmessage = (event) => {
        clearTimeout(timeout);
        const tokenPacket = decodePacket(event.data);
        assert.equal(tokenPacket.msgType, MSG_TYPES.TOKEN, "Phone A expects token packet");
        resolve(tokenPacket.payload.token);
      };
      phoneA.channel.send(activationPacket);
    });

    const stepEnd = performance.now();
    const tokenLatency = stepEnd - stepStart;
    phoneA.metrics.tokenLatencies.push(tokenLatency);

    phoneA.context.push(receivedToken);

    console.log(
      `  [Step ${(step + 1).toString().padStart(2)}] Token: ${receivedToken.toString().padStart(3)} (${reprByte(receivedToken)}) | ` +
      `Phone A: ${aComputeTime.toFixed(1)}ms | ` +
      `Phone B: ${phoneB.metrics.bComputeTimes[step].toFixed(1)}ms | ` +
      `Round-Trip: ${tokenLatency.toFixed(1)}ms | ` +
      `Payload: ${(activationPacket.byteLength / 1024).toFixed(1)}KB`
    );

    assert.equal(
      receivedToken,
      expectedSequence[4 + step],
      `Mismatch at step ${step + 1}: expected ${expectedSequence[4 + step]}, got ${receivedToken}`
    );
  }

  const totalGenEnd = performance.now();
  const totalGenTime = totalGenEnd - totalGenStart;

  console.log("\n----------------------------------------------------------------------");
  console.log("DISTRIBUTED INFERENCE BENCHMARK REPORT");
  console.log("----------------------------------------------------------------------");
  console.log(`Prompt:                     "${Buffer.from([77,69,83,72]).toString('utf-8')}" (bytes: [77, 69, 83, 72])`);
  console.log(`Generated:                  ${JSON.stringify(phoneA.context)}`);
  console.log(`Total generated tokens:     ${stepsToGenerate}`);
  console.log(`Total generation time:      ${totalGenTime.toFixed(2)} ms`);
  console.log(`Throughput:                 ${(stepsToGenerate / (totalGenTime / 1000)).toFixed(2)} tokens/sec`);
  console.log(`Mean Phone A compute:       ${mean(phoneA.metrics.aComputeTimes).toFixed(2)} ms/step`);
  console.log(`Mean Phone B compute:       ${mean(phoneB.metrics.bComputeTimes).toFixed(2)} ms/step`);
  console.log(`Mean Round-Trip Latency:    ${mean(phoneA.metrics.tokenLatencies).toFixed(2)} ms/step`);
  console.log(`Total bytes sent:           ${phoneA.metrics.bytesTransferred.toLocaleString()} bytes (${(phoneA.metrics.bytesTransferred / 1024).toFixed(2)} KB)`);
  console.log("----------------------------------------------------------------------");

  assert.deepEqual(phoneA.context, expectedSequence);
  console.log("✓ Distributed output MATCHES reference implementation bit-for-bit across all 10 tokens!");
  console.log("======================================================================");
  console.log("TEST D PASSED SUCCESSFULLY!");
  console.log("======================================================================");
}

function reprByte(b) {
  if (b >= 32 && b <= 126) return `'${String.fromCharCode(b)}'`;
  return `0x${b.toString(16).padStart(2, "0")}`;
}

function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

runDistributedInferenceTest().catch((err) => {
  console.error("Test D failed:", err);
  process.exit(1);
});
