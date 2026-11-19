/**
 * Stage 7 — the actual measurement loops. One function per kind of benchmark; all of them
 *   - run `warmup` discarded iterations first (V8 compiles hot functions after they have run a few times),
 *   - then `iterations` measured iterations,
 *   - catch every failure, store it in the Recorder's error list and keep it OUT of the latency statistics,
 *   - verify that the produced tokens are the expected ones (a fast wrong answer is a failure, not a result).
 */
"use strict";
const P = require("../../protocol/mesh_protocol");
const { PipelineA, PipelineB } = require("../../engine/mesh_pipeline");
const { Recorder } = require("./stats");
const { Instrument, PHASES } = require("./instrument");
const { LoopbackLink } = require("./loopback");
const { PROMPT, GEN_TOKENS, REFERENCE_GENERATION } = require("./model");
const now = () => performance.now();
const D_MODEL = 384;

/* ---------------------------------- local ---------------------------------- */

/** One forward pass of the complete model on one device (no serialization, no transport). */
function localForwardRow(model, tokens) {
  const t0 = now(); const r = model.full.forward(tokens); const ms = now() - t0;
  return { row: { totalMs: ms }, token: r.nextByte };
}

/** A whole greedy generation, exactly like MeshFullModel.generate(), timed per token. */
function localGenerateRow(model, prompt, newTokens) {
  const toks = Array.from(prompt); const perToken = [];
  const t0 = now();
  for (let i = 0; i < newTokens; i++) { const a = now(); const { nextByte } = model.full.forward(toks); perToken.push(now() - a); toks.push(nextByte); }
  const totalMs = now() - t0;
  return { row: { totalMs, msPerToken: totalMs / newTokens, tokensPerSec: newTokens / (totalMs / 1000), lastTokenMs: perToken[perToken.length - 1] }, tokens: toks };
}

function runLocal(model, workload, { warmup, iterations }) {
  const rec = new Recorder();
  for (let i = 0; i < warmup + iterations; i++) {
    try {
      if (workload.kind === "forward") {
        const tokens = workload.tokens;
        const { row, token } = localForwardRow(model, tokens);
        if (workload.expectedToken !== undefined && token !== workload.expectedToken) throw Object.assign(new Error(`token ${token} != expected ${workload.expectedToken}`), { code: "OUTPUT_MISMATCH" });
        if (i >= warmup) rec.ok(row);
      } else {
        const { row, tokens } = localGenerateRow(model, workload.prompt, workload.newTokens);
        if (workload.expectedTokens && tokens.join() !== workload.expectedTokens.join()) throw Object.assign(new Error("generated sequence differs from the reference"), { code: "OUTPUT_MISMATCH" });
        if (i >= warmup) rec.ok(row);
      }
    } catch (e) { if (i >= warmup) rec.fail(e, { iteration: i - warmup }); }
  }
  return rec;
}

/* -------------------------------- distributed -------------------------------- */

/** Build A and B joined by a loopback link, finish the HELLO handshake, and install the timing wrappers. */
async function makeRig(model, { maxMessageSize = 262144, instrument = true } = {}) {
  const link = new LoopbackLink(maxMessageSize);
  const tA = link.transport("AtoB"), tB = link.transport("BtoA");
  const pipeA = new PipelineA({ engine: model.partitionA, fingerprint: model.fingerprint, transport: tA, timeoutMs: 120000 });
  const pipeB = new PipelineB({ engine: model.partitionB, fingerprint: model.fingerprint, transport: tB, timeoutMs: 120000 });
  pipeA.reassembler.__owner = "A"; pipeB.reassembler.__owner = "B";
  link.toA = (m) => pipeA.handleMessage(m); link.toB = (m) => pipeB.handleMessage(m);
  const inst = new Instrument();
  if (instrument) inst.install({ engineA: model.partitionA, engineB: model.partitionB, transportA: tA, transportB: tB });
  await pipeA.sendHello(); await pipeB.sendHello(); await pipeA.waitReady(); await pipeB.waitReady();
  return { link, pipeA, pipeB, inst, dispose() { inst.uninstall(); pipeA.dispose("rig done"); pipeB.dispose("rig done"); } };
}

/** One distributed request. Returns a flat metrics row. Throws on failure (caller records it). */
async function distributedRequest(rig, tokens) {
  rig.inst.begin();
  const t0 = now();
  const r = await rig.pipeA.infer(tokens);
  const totalMs = now() - t0;
  const { spans, counts } = rig.inst.end();
  const attributed = PHASES.reduce((s, p) => s + spans[p], 0);
  const row = { totalMs, ...spans, attributedMs: attributed, unattributedMs: totalMs - attributed,
    serializeMs: spans.aSerialize + spans.bSerializeReply, chunkMs: spans.aChunk + spans.bChunkReply, reassembleMs: spans.bReassemble + spans.aReassembleReply,
    decodeMs: spans.bDecode + spans.aDecodeReply, sendCallMs: spans.aSend + spans.bSend, computeMs: spans.aCompute + spans.bCompute,
    protocolMs: totalMs - spans.aCompute - spans.bCompute,        // everything that is not model compute
    activationPacketBytes: r.activationBytes, frames: r.frames, framesBA: counts.framesBA,
    // the pipeline's own self-reported numbers, kept only to cross-check the wrappers
    pipeComputeMsA: r.computeMsA, pipeComputeMsB: r.computeMsB, pipeRttMs: r.rttMs, pipeTotalMs: r.totalMs };
  return { row, token: r.token };
}

async function runDistributed(model, workload, { warmup, iterations, maxMessageSize }) {
  const rig = await makeRig(model, { maxMessageSize });
  const rec = new Recorder();
  try {
    for (let i = 0; i < warmup + iterations; i++) {
      const measured = i >= warmup;
      try {
        if (workload.kind === "forward") {
          const { row, token } = await distributedRequest(rig, workload.tokens);
          if (workload.expectedToken !== undefined && token !== workload.expectedToken) throw Object.assign(new Error(`token ${token} != expected ${workload.expectedToken}`), { code: "OUTPUT_MISMATCH" });
          if (measured) rec.ok(row);
        } else {
          const toks = Array.from(workload.prompt); const rows = [];
          const t0 = now();
          for (let k = 0; k < workload.newTokens; k++) { const { row, token } = await distributedRequest(rig, toks); rows.push(row); toks.push(token); }
          const totalMs = now() - t0;
          if (workload.expectedTokens && toks.join() !== workload.expectedTokens.join()) throw Object.assign(new Error("generated sequence differs from the reference"), { code: "OUTPUT_MISMATCH" });
          const sum = (k) => rows.reduce((s, r) => s + r[k], 0);
          if (measured) rec.ok({ totalMs, msPerToken: totalMs / workload.newTokens, tokensPerSec: workload.newTokens / (totalMs / 1000),
            computeMs: sum("computeMs"), protocolMs: sum("protocolMs"), serializeMs: sum("serializeMs"), chunkMs: sum("chunkMs"), reassembleMs: sum("reassembleMs"),
            decodeMs: sum("decodeMs"), sendCallMs: sum("sendCallMs"), unattributedMs: sum("unattributedMs"), bytesSentAB: sum("activationPacketBytes"), requests: rows.length });
        }
      } catch (e) { if (measured) rec.fail(e, { iteration: i - warmup }); else throw e; }   // a failing WARM-UP means the rig itself is broken: stop loudly
    }
  } finally { rig.dispose(); }
  return rec;
}

module.exports = { runLocal, runDistributed, makeRig, distributedRequest, PHASES, PROMPT, GEN_TOKENS, REFERENCE_GENERATION, D_MODEL };
