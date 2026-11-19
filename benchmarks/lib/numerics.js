/**
 * Stage 7 — numerical correctness gate. Run before AND after every optimization.
 *
 * Measures (never assumes):
 *   - local JS vs PyTorch fixture: max |difference| of the A-activation and of the logits, and argmax agreement
 *   - distributed JS vs local JS: bit-for-bit equality of logits through the real pipeline (encode -> chunk -> decode -> B)
 *   - serialization fidelity: decodePacket(encodeActivation(x)).payload is bit-identical to x
 *   - logitsDigest: Adler-32 over the raw logit bytes of all fixture cases. If this number is unchanged after an optimization, the
 *     optimized engine produced BIT-IDENTICAL logits. If it changes, maxAbsPyTorchLogits tells how far the values moved.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const P = require("../../protocol/mesh_protocol");
const { makeRig } = require("./runners");
const FIXTURE = path.join(__dirname, "..", "..", "tests", "fixtures", "stage5_reference.json");

const f32 = (b64) => { const b = Buffer.from(b64, "base64"); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const bitEqual = (a, b) => { if (a.length !== b.length) return false; const x = new Uint32Array(a.buffer, a.byteOffset, a.length), y = new Uint32Array(b.buffer, b.byteOffset, b.length); for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false; return true; };
const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

async function numericalCheck(model, { maxT = Infinity } = {}) {
  const fx = JSON.parse(fs.readFileSync(FIXTURE, "utf-8"));
  const cases = fx.cases.filter((c) => c.T <= maxT);
  const rig = await makeRig(model, { instrument: false });
  const res = { tolerance: fx.tolerance, cases: [], maxAbsPyTorchActivation: 0, maxAbsPyTorchLogits: 0, allArgmaxMatchPyTorch: true,
    distributedEqualsLocalBitForBit: true, serializationBitExact: true };
  const digestParts = [];
  try {
    for (const c of cases) {
      const act = model.partitionA.forward(c.ids);
      const local = model.partitionB.forward(act, c.T);
      const pyAct = f32(c.act_b64), pyLogits = f32(c.logits_b64);
      const dAct = maxAbs(act, pyAct), dLog = maxAbs(local.logits, pyLogits);
      const dec = P.decodePacket(P.encodeActivation({ sessionId: 1n, seq: 1, activations: act, shape: [1, c.T, 384] }));
      const serOk = bitEqual(dec.payload, act);
      const dist = await rig.pipeA.infer(c.ids, { wantLogits: true });
      const distOk = dist.token === local.nextByte && bitEqual(dist.logits, local.logits);
      res.cases.push({ name: c.name, T: c.T, maxAbsActivationVsPyTorch: dAct, maxAbsLogitsVsPyTorch: dLog, argmaxMatchesPyTorch: local.nextByte === c.argmax, serializationBitExact: serOk, distributedBitEqualLocal: distOk });
      res.maxAbsPyTorchActivation = Math.max(res.maxAbsPyTorchActivation, dAct);
      res.maxAbsPyTorchLogits = Math.max(res.maxAbsPyTorchLogits, dLog);
      if (local.nextByte !== c.argmax) res.allArgmaxMatchPyTorch = false;
      if (!distOk) res.distributedEqualsLocalBitForBit = false;
      if (!serOk) res.serializationBitExact = false;
      digestParts.push(new Uint8Array(local.logits.buffer.slice(local.logits.byteOffset, local.logits.byteOffset + local.logits.byteLength)));
    }
  } finally { rig.dispose(); }
  const all = new Uint8Array(digestParts.reduce((s, p) => s + p.length, 0)); let o = 0; for (const p of digestParts) { all.set(p, o); o += p.length; }
  res.logitsDigestAdler32 = P.adler32(all, 0, all.length);
  res.passed = res.maxAbsPyTorchActivation <= res.tolerance && res.maxAbsPyTorchLogits <= res.tolerance && res.allArgmaxMatchPyTorch && res.distributedEqualsLocalBitForBit && res.serializationBitExact;
  return res;
}
module.exports = { numericalCheck, bitEqual, maxAbs };
