/**
 * MESH Performance Instrumentation & Benchmark Suite
 * Generates actual measurements for:
 * - Weight loading time per device
 * - Layer execution time on Phone A (layers 0-2)
 * - Layer execution time on Phone B (layers 3-5)
 * - Activation serialization / deserialization time
 * - Token generation latency
 * - Total generation time
 * - Generated tokens per second
 * - Total bytes transferred over wire
 */

"use strict";

const fs = require("fs");
const path = require("path");
const { MeshPartitionA, MeshPartitionB } = require("../engine/inference_engine");
const {
  encodeActivation,
  encodeToken,
  decodePacket,
} = require("../protocol/mesh_protocol");

const WEIGHTS_DIR = path.join(__dirname, "..", "room-server", "public", "weights");

function runBenchmark() {
  console.log("======================================================================");
  console.log("MESH COMPREHENSIVE PERFORMANCE BENCHMARK");
  console.log("======================================================================");

  // 1. Weight Loading Measurement
  const manifest = JSON.parse(fs.readFileSync(path.join(WEIGHTS_DIR, "weights_manifest.json"), "utf-8"));

  const tLoadA0 = performance.now();
  const partABuf = fs.readFileSync(path.join(WEIGHTS_DIR, "partition_a.bin"));
  const abA = partABuf.buffer.slice(partABuf.byteOffset, partABuf.byteOffset + partABuf.byteLength);
  const engineA = MeshPartitionA.fromBuffer(abA, manifest);
  const tLoadA1 = performance.now();
  const loadTimeA = tLoadA1 - tLoadA0;

  const tLoadB0 = performance.now();
  const partBBuf = fs.readFileSync(path.join(WEIGHTS_DIR, "partition_b.bin"));
  const abB = partBBuf.buffer.slice(partBBuf.byteOffset, partBBuf.byteOffset + partBBuf.byteLength);
  const engineB = MeshPartitionB.fromBuffer(abB, manifest);
  const tLoadB1 = performance.now();
  const loadTimeB = tLoadB1 - tLoadB0;

  console.log(`Phone A Weight Loading: ${loadTimeA.toFixed(1)} ms (${(partABuf.length / 1024 / 1024).toFixed(2)} MB, 5,618,304 params)`);
  console.log(`Phone B Weight Loading: ${loadTimeB.toFixed(1)} ms (${(partBBuf.length / 1024 / 1024).toFixed(2)} MB, 5,422,464 params)`);

  // 2. Multi-Step Generation Benchmark
  const prompt = "MESH distributed AI benchmark test";
  const promptBytes = Array.from(Buffer.from(prompt, "utf-8"));
  const numSteps = 20;

  console.log(`\nRunning ${numSteps}-token distributed benchmark on prompt "${prompt}" (${promptBytes.length} bytes)...`);

  const currentTokens = [...promptBytes];
  let totalBytesSent = 0;
  const aTimes = [];
  const bTimes = [];
  const serTimes = [];
  const deserTimes = [];
  const stepLatencies = [];

  const benchStart = performance.now();

  for (let s = 0; s < numSteps; s++) {
    const sStart = performance.now();

    // Phone A: Layers 0-2
    const a0 = performance.now();
    const act = engineA.forward(currentTokens);
    const a1 = performance.now();
    aTimes.push(a1 - a0);

    // Serialization
    const ser0 = performance.now();
    const packet = encodeActivation({
      sessionId: 42n,
      seq: s + 1,
      activations: act,
      shape: [1, currentTokens.length, 384, 0],
    });
    const ser1 = performance.now();
    serTimes.push(ser1 - ser0);
    totalBytesSent += packet.byteLength;

    // Deserialization on Phone B
    const deser0 = performance.now();
    const decoded = decodePacket(packet);
    const deser1 = performance.now();
    deserTimes.push(deser1 - deser0);

    // Phone B: Layers 3-5 + LM Head
    const b0 = performance.now();
    const { nextByte } = engineB.forward(decoded.payload, currentTokens.length);
    const b1 = performance.now();
    bTimes.push(b1 - b0);

    // Token return packet
    const tokenPacket = encodeToken({
      sessionId: 42n,
      seq: s + 1,
      token: nextByte,
      isEos: false,
    });
    totalBytesSent += tokenPacket.byteLength;

    const sEnd = performance.now();
    stepLatencies.push(sEnd - sStart);
    currentTokens.push(nextByte);
  }

  const benchEnd = performance.now();
  const totalDuration = benchEnd - benchStart;

  const avg = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
  const p95 = (arr) => {
    const s = [...arr].sort((a, b) => a - b);
    return s[Math.floor(s.length * 0.95)];
  };

  console.log("\n======================================================================");
  console.log("ACTUAL BENCHMARK MEASUREMENTS SUMMARY");
  console.log("======================================================================");
  console.log(`Hardware / Runtime:        Node.js ${process.version} (${process.arch}) CPU`);
  console.log(`Model Architecture:        Byte-Level Micro-Transformer (6 Layers, D=384, H=6, Dff=1536)`);
  console.log(`Total Model Parameters:    11,040,768 (FP32: 44.16 MB)`);
  console.log(`Phone A Partition:         Layers 0-2 (5,618,304 params / 22.47 MB FP32)`);
  console.log(`Phone B Partition:         Layers 3-5 + Head (5,422,464 params / 21.69 MB FP32)`);
  console.log("----------------------------------------------------------------------");
  console.log(`Tokens Generated:          ${numSteps}`);
  console.log(`Total Generation Time:     ${totalDuration.toFixed(2)} ms (${(totalDuration / 1000).toFixed(2)} s)`);
  console.log(`Throughput:                ${(numSteps / (totalDuration / 1000)).toFixed(2)} tokens/sec`);
  console.log(`Phone A Compute Mean:      ${avg(aTimes).toFixed(2)} ms/token`);
  console.log(`Phone B Compute Mean:      ${avg(bTimes).toFixed(2)} ms/token`);
  console.log(`Serialization Mean:        ${avg(serTimes).toFixed(3)} ms/token`);
  console.log(`Deserialization Mean:      ${avg(deserTimes).toFixed(3)} ms/token`);
  console.log(`Total Token Latency Mean:  ${avg(stepLatencies).toFixed(2)} ms/token`);
  console.log(`Total Token Latency P95:   ${p95(stepLatencies).toFixed(2)} ms/token`);
  console.log(`Total Wire Bytes Sent:     ${totalBytesSent.toLocaleString()} bytes (${(totalBytesSent / 1024).toFixed(2)} KB)`);
  console.log(`Avg Wire Bandwidth:        ${((totalBytesSent / (totalDuration / 1000)) / 1024).toFixed(2)} KB/sec`);
  console.log("======================================================================");

  // Write benchmark results to JSON file
  const report = {
    timestamp: new Date().toISOString(),
    runtime: `Node.js ${process.version}`,
    arch: process.arch,
    platform: process.platform,
    model: {
      parameters: 11040768,
      layers: 6,
      d_model: 384,
      d_ff: 1536,
      heads: 6,
      partition_a_params: 5618304,
      partition_b_params: 5422464,
    },
    measurements: {
      tokensGenerated: numSteps,
      totalDurationMs: totalDuration,
      throughputTokensPerSec: numSteps / (totalDuration / 1000),
      phoneAComputeMs: { mean: avg(aTimes), min: Math.min(...aTimes), max: Math.max(...aTimes) },
      phoneBComputeMs: { mean: avg(bTimes), min: Math.min(...bTimes), max: Math.max(...bTimes) },
      serializationMs: { mean: avg(serTimes) },
      deserializationMs: { mean: avg(deserTimes) },
      tokenLatencyMs: { mean: avg(stepLatencies), p95: p95(stepLatencies) },
      totalBytesSent,
      avgBandwidthKBps: (totalBytesSent / (totalDuration / 1000)) / 1024,
    },
    generatedTokens: currentTokens,
  };

  const reportPath = path.join(__dirname, "..", "BENCHMARK_RESULTS.json");
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`Full report saved to: ${reportPath}`);
}

runBenchmark();
