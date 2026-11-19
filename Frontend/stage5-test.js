"use strict";
/* MINIMOL Stage 5: browser validation helper. Loaded after main.js and dc-test.js, uses main.js globals (S, CONFIG, libs, note).

   Console usage:
     await meshStage5.validate()     // everything that can run here; two-device checks are SKIPPED (never "passed") unless a live pipeline exists
     await meshStage5.local()        // only the checks that need NO peer
     await meshStage5.twoDevice()    // only the checks that need a connected peer (run on Device A; Device B can run the connection checks)
   Options: validate({ cases: ["MESH", "sentence31"] }) limits which Stage 5 reference cases are used.

   Two kinds of checks, reported separately:
   LOCAL (one browser instance, nothing leaves this tab):
     real weights downloaded from the server, real engine, real protocol and pipeline code, but the "DataChannel" is an in-memory
     stand-in. This proves the BROWSER's JS engine/protocol/pipeline are correct. It does NOT prove anything about WebRTC.
   TWO-DEVICE (needs two tabs or two devices connected through the room-server):
     the live pipeline (S.pipe) over the real DataChannel. Only these prove the integration end to end.

   Each check is PASS, FAIL or SKIP (with the reason). Nothing is ever reported as PASS without actually running. */
(() => {
  const FIXTURE_URL = "/fixtures/stage5_reference.json";
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const f32 = (b64) => { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new Float32Array(u.buffer); };
  const maxAbs = (a, b) => { if (a.length !== b.length) return Infinity; let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
  const bitEqual = (a, b) => { if (a.length !== b.length) return false; const x = new Uint32Array(a.buffer, a.byteOffset, a.length), y = new Uint32Array(b.buffer, b.byteOffset, b.length); for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false; return true; };
  const expectCode = async (fn, code) => {
    try { await fn(); } catch (e) { if (e.code !== code) throw new Error(`expected error code ${code}, got ${e.code || "none"}: ${e.message}`); return; }
    throw new Error(`expected error ${code}, but nothing failed`);
  };
  const need = (cond, msg) => { if (!cond) throw new Error(msg); };
  class Skip extends Error {}
  const skip = (why) => { throw new Skip(why); };

  /* ---------- tiny test runner ---------- */
  function runner() {
    const rows = [];
    return {
      rows,
      async check(group, name, fn) {
        const t0 = performance.now(); let status = "PASS", detail = "";
        try { const d = await fn(); if (typeof d === "string") detail = d; }
        catch (e) { if (e instanceof Skip) { status = "SKIP"; detail = e.message; } else { status = "FAIL"; detail = e.message; } }
        rows.push({ group, check: name, status, ms: Math.round(performance.now() - t0), detail });
        console.log(`[meshStage5] ${status.padEnd(4)} ${group} / ${name}${detail ? "  -  " + detail : ""}`);
      },
    };
  }
  function summarise(rows, title) {
    const n = (s) => rows.filter((r) => r.status === s).length;
    console.log(`\n[meshStage5] ${title}`); console.table(rows);
    const s = { pass: n("PASS"), fail: n("FAIL"), skip: n("SKIP"), rows };
    console.log(`[meshStage5] ${title}: ${s.pass} passed, ${s.fail} failed, ${s.skip} skipped` + (s.skip ? " (SKIPPED checks were NOT run and prove nothing)" : ""));
    return s;
  }

  /* ---------- shared data (downloaded once per page load) ---------- */
  let cache = null;
  async function data() {
    if (cache) return cache;
    await libs();                                                                         // main.js: loads MeshProtocol / MeshInference / MeshPipeline if missing
    const get = async (url, kind) => { const r = await fetch(url, { cache: "no-store" }); if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r[kind](); };
    const [manifest, fixture, bufA, bufB] = await Promise.all([get(CONFIG.paths.manifest, "json"), get(FIXTURE_URL, "json"), get(CONFIG.paths.a, "arrayBuffer"), get(CONFIG.paths.b, "arrayBuffer")]);
    for (const c of fixture.cases) { c.act = f32(c.act_b64); c.logits = f32(c.logits_b64); }
    cache = { manifest, fixture, bufA, bufB };
    return cache;
  }
  const pickCases = (fixture, names) => (names ? fixture.cases.filter((c) => names.includes(c.name)) : fixture.cases);

  /* ---------- in-memory stand-in for a DataChannel pair (enforces the message-size limit like a browser) ---------- */
  function loopback(max) {
    const l = { max, biggest: 0, frames: 0, log: { AtoB: [], BtoA: [] }, toA: null, toB: null };
    const side = (dir) => ({ maxMessageSize: () => max, send: (buf) => {
      if (buf.byteLength > max) throw new RangeError(`message of ${buf.byteLength} bytes exceeds the limit ${max}`);
      l.biggest = Math.max(l.biggest, buf.byteLength); l.frames++; l.log[dir].push(buf.slice(0));
      const copy = buf.slice(0); setTimeout(() => (dir === "AtoB" ? l.toB : l.toA)(copy), 0);
    } });
    l.a = side("AtoB"); l.b = side("BtoA");
    return l;
  }
  async function until(cond, ms = 5000) { const t0 = performance.now(); while (!cond()) { if (performance.now() - t0 > ms) throw new Error("condition not met in " + ms + " ms"); await sleep(5); } }

  /* =========================================================================================================== */
  /*  LOCAL checks: one browser instance                                                                         */
  /* =========================================================================================================== */
  async function local({ cases: only } = {}) {
    const R = runner(), G = "LOCAL (one browser, in-memory transport, no WebRTC)";
    let D;
    await R.check(G, "scripts + weights + reference fixture download", async () => {
      D = await data();
      need(window.MeshProtocol && window.MeshInference && window.MeshPipeline, "MeshProtocol / MeshInference / MeshPipeline missing");
      return `${D.fixture.cases.length} reference cases, PyTorch ${D.fixture.torch_version}, tolerance ${D.fixture.tolerance}`;
    });
    if (!D) return summarise(R.rows, "local checks");
    const MP = MeshProtocol, MI = MeshInference, PL = MeshPipeline, { manifest, fixture } = D, TOL = fixture.tolerance;
    const cases = pickCases(fixture, only);
    let engA, engB, fp;
    const locals = {};   // case name -> { act, logits } computed by calling the engines directly (no transfer)

    await R.check(G, "weights: Adler-32 + size match the manifest (A and B)", () => {
      const a = PL.verifyPartitionBuffer(D.bufA, manifest, "a"), b = PL.verifyPartitionBuffer(D.bufB, manifest, "b");
      return `A=${a} B=${b}`;
    });
    await R.check(G, "weights: manifest checksums equal the PyTorch fixture's", () => {
      need(manifest.partition_a.checksum_adler32 === fixture.partition_a_adler32, "partition A checksum differs from the fixture: weights and fixture are from different exports");
      need(manifest.partition_b.checksum_adler32 === fixture.partition_b_adler32, "partition B checksum differs from the fixture");
    });
    await R.check(G, "weights: a flipped byte, a short file and a manifest without checksums are all REJECTED", async () => {
      const bad = D.bufA.slice(0); new Uint8Array(bad)[1000] ^= 1;
      await expectCode(() => PL.verifyPartitionBuffer(bad, manifest, "a"), "MODEL_MISMATCH");
      await expectCode(() => PL.verifyPartitionBuffer(D.bufA.slice(0, D.bufA.byteLength - 4), manifest, "a"), "MODEL_MISMATCH");
      const noSum = JSON.parse(JSON.stringify(manifest)); delete noSum.partition_a.checksum_adler32;
      await expectCode(() => PL.verifyPartitionBuffer(D.bufA, noSum, "a"), "MODEL_MISMATCH");
    });
    await R.check(G, "model fingerprint: built from the manifest, identical when rebuilt, different if a checksum differs", () => {
      fp = PL.modelFingerprint(manifest);
      need(fp === PL.modelFingerprint(JSON.parse(JSON.stringify(manifest))), "fingerprint is not deterministic");
      const other = JSON.parse(JSON.stringify(manifest)); other.partition_b.checksum_adler32 ^= 1;
      need(PL.modelFingerprint(other) !== fp, "fingerprint ignores partition B's checksum");
      return fp.length > 60 ? fp.slice(0, 60) + "…" : fp;
    });
    await R.check(G, "engines: build partition A and B from the verified buffers", () => {
      engA = MI.MeshPartitionA.fromBuffer(D.bufA, manifest); engB = MI.MeshPartitionB.fromBuffer(D.bufB, manifest);
    });
    if (!engA) return summarise(R.rows, "local checks");

    for (const c of cases) {
      await R.check(G, `reference "${c.name}" (T=${c.T}): A's activation, tensor packet, B's logits vs PyTorch`, () => {
        const act = engA.forward(c.ids);
        const dA = maxAbs(act, c.act);
        need(dA <= TOL, `activation differs from PyTorch by ${dA.toExponential(2)} (> ${TOL})`);
        // Tensor validation: the exact functions Device B runs on arrival.
        const pkt = MP.encodeActivation({ sessionId: 1n, seq: 1, activations: act, shape: [1, c.T, 384] });
        const dec = MP.decodePacket(pkt); MP.validateActivation(dec, { dModel: MI.D_MODEL, maxSeqLen: MI.MAX_SEQ_LEN });
        need(bitEqual(dec.payload, act), "activation changed while being serialised and decoded");
        const r = engB.forward(dec.payload, c.T);
        const dL = maxAbs(r.logits, c.logits);
        need(dL <= TOL, `logits differ from PyTorch by ${dL.toExponential(2)} (> ${TOL})`);
        need(r.nextByte === c.argmax, `next byte ${r.nextByte} != PyTorch argmax ${c.argmax}`);
        locals[c.name] = { logits: r.logits, token: r.nextByte };
        return `act diff ${dA.toExponential(2)}, logits diff ${dL.toExponential(2)}, token ${r.nextByte}`;
      });
    }

    await R.check(G, "tensor validation: NaN, wrong rank, wrong hidden size, wrong dtype, corrupted bytes are all rejected", async () => {
      const x = new Float32Array(2 * 384).fill(0.5), good = () => MP.decodePacket(MP.encodeActivation({ sessionId: 1n, seq: 1, activations: x, shape: [1, 2, 384] }));
      const nan = x.slice(); nan[5] = NaN;
      await expectCode(() => MP.validateActivation(MP.decodePacket(MP.encodeActivation({ sessionId: 1n, seq: 1, activations: nan, shape: [1, 2, 384] }))), "NON_FINITE");
      await expectCode(() => MP.validateActivation(MP.decodePacket(MP.encodeActivation({ sessionId: 1n, seq: 1, activations: x, shape: [2, 384] }))), "BAD_SHAPE");
      await expectCode(() => MP.validateActivation(MP.decodePacket(MP.encodeActivation({ sessionId: 1n, seq: 1, activations: new Float32Array(2 * 100), shape: [1, 2, 100] }))), "BAD_SHAPE");
      const d = good(); d.dataType = MP.DATA_TYPES.INT32;
      await expectCode(() => MP.validateActivation(d), "BAD_DTYPE");
      const pkt = MP.encodeActivation({ sessionId: 1n, seq: 1, activations: x, shape: [1, 2, 384] }); new Uint8Array(pkt)[100] ^= 1;
      try { MP.decodePacket(pkt); throw new Error("corrupted packet was accepted"); } catch (e) { need(/checksum/i.test(e.message), "corruption not caught by the checksum: " + e.message); }
    });

    // Loopback pipeline: real PipelineA + PipelineB, real chunking, but an in-memory transport.
    const runLoop = async (c, max) => {
      const L = loopback(max), evB = [];
      const pa = new PL.PipelineA({ engine: engA, fingerprint: fp, transport: L.a, timeoutMs: 600000 });
      const pb = new PL.PipelineB({ engine: engB, fingerprint: fp, transport: L.b, timeoutMs: 600000, onEvent: (t, i) => evB.push({ t, ...i }) });
      L.toA = (m) => pa.handleMessage(m); L.toB = (m) => pb.handleMessage(m);
      await pa.sendHello(); await pb.sendHello(); await until(() => pa.peer && pb.peer);
      need(pa.peer.ok && pb.peer.ok, "HELLO handshake not accepted: " + (pa.peer.reason || pb.peer.reason));
      return { L, pa, pb, evB };
    };
    const loopCases = [["MESH", 262144], ["sentence31", 16384], ["long200", 262144]].filter(([n]) => cases.some((c) => c.name === n));
    for (const [name, max] of loopCases) {
      const c = cases.find((x) => x.name === name);
      await R.check(G, `loopback pipeline "${name}", message limit ${max}: transfer, session/seq match, result equals direct call and PyTorch`, async () => {
        if (!locals[name]) skip("the direct-engine check for this case failed, so there is nothing to compare with");
        const { L, pa, pb } = await runLoop(c, max);
        const r = await pa.infer(c.ids, { wantLogits: true });
        need(L.biggest <= max, `a frame of ${L.biggest} bytes exceeded the limit ${max}`);
        need(r.token === c.argmax, `token ${r.token} != PyTorch argmax ${c.argmax}`);
        need(bitEqual(r.logits, locals[name].logits), "logits through the pipeline differ from calling the engines directly: transfer is not lossless");
        need(r.seq === 1 && /^\d+$/.test(r.sessionId) && r.sessionId !== "0", "reply is not tied to a session/seq");
        need(pb.stats.processed === 1, "Device B processed " + pb.stats.processed + " activations, expected exactly 1");
        const expectChunks = r.activationBytes > max;
        need(expectChunks ? r.frames > 1 : r.frames === 1, `${r.activationBytes}-byte activation was sent in ${r.frames} frame(s) with limit ${max}`);
        return `${r.activationBytes} B in ${r.frames} frame(s), logits bit-identical to direct call, A ${r.computeMsA.toFixed(0)} ms / B ${r.computeMsB.toFixed(0)} ms`;
      });
    }
    if (cases.some((c) => c.name === "MESH")) {
      const c = cases.find((x) => x.name === "MESH");
      await R.check(G, "loopback pipeline: wrong fingerprint refused; duplicate activation refused; reply for another session ignored", async () => {
        // 1. fingerprint mismatch
        const L1 = loopback(262144), pa1 = new PL.PipelineA({ engine: engA, fingerprint: fp, transport: L1.a, timeoutMs: 600000 }), pb1 = new PL.PipelineB({ engine: engB, fingerprint: fp + "x", transport: L1.b, timeoutMs: 600000 });
        L1.toA = (m) => pa1.handleMessage(m); L1.toB = (m) => pb1.handleMessage(m);
        await pa1.sendHello(); await pb1.sendHello(); await until(() => pa1.peer && pb1.peer);
        await expectCode(() => pa1.infer(c.ids), "MODEL_MISMATCH");
        // 2. duplicate: replay the exact activation packet that already went A -> B
        const { L, pa, pb } = await runLoop(c, 262144);
        await pa.infer(c.ids);
        const replay = L.log.AtoB.find((b) => MP.decodePacket(b).msgType === MP.MSG_TYPES.ACTIVATION);
        const before = L.log.BtoA.length; await pb.handleMessage(replay.slice(0));
        await until(() => L.log.BtoA.length > before);
        const rej = MP.decodePacket(L.log.BtoA[L.log.BtoA.length - 1]);
        need(rej.msgType === MP.MSG_TYPES.ERROR && rej.payload.code === "DUPLICATE", "duplicate was not answered with an ERROR/DUPLICATE packet");
        need(pb.stats.processed === 1, "duplicate was computed again");
        // 3. a TOKEN reply for a session A never asked about must not settle anything
        const ghost = MP.encodeToken({ sessionId: 999n, seq: 1, token: 7 });
        await pa.handleMessage(ghost);
        need(pa.pending === null, "A is unexpectedly waiting after a stray reply");
      });
    }
    return summarise(R.rows, "local checks");
  }

  /* =========================================================================================================== */
  /*  TWO-DEVICE checks: live pipeline over the real DataChannel                                                  */
  /* =========================================================================================================== */
  async function twoDevice({ cases: only } = {}) {
    const R = runner(), G = "TWO-DEVICE (live WebRTC DataChannel)";
    const live = () => S.pipe && S.dc && S.dc.readyState === "open";
    await R.check(G, "pipeline initialised on this device (weights verified AND DataChannel open)", () => {
      if (S.demo) skip("dev demo mode has no peer");
      if (!S.dc || S.dc.readyState !== "open") skip("no open DataChannel: connect two devices/tabs first");
      if (S.loadError) throw new Error(S.loadError);
      need(S.pipe, `DataChannel is open but the pipeline is not initialised (weights: ${S.load})`);
      return `role ${S.role}, weights ${S.load}`;
    });
    await R.check(G, "peer HELLO received and the model fingerprint matches", async () => {
      if (!live()) skip("no live pipeline");
      await S.pipe.waitReady(10000);
      need(S.pipe.peer && S.pipe.peer.ok, "peer not verified");
      return "peer confirmed it has loaded the same model/weights";
    });
    await R.check(G, "transport: negotiated message limit and send-queue settings", () => {
      if (!live()) skip("no live pipeline");
      const m = S.pc && S.pc.sctp && S.pc.sctp.maxMessageSize;
      need(S.dc.bufferedAmountLowThreshold === CONFIG.bufferLow, "bufferedAmountLowThreshold not set");
      return `pc.sctp.maxMessageSize=${m === undefined ? "unavailable (fallback " + CONFIG.fallbackMaxMessage + ")" : m}, using ${dcTransport.maxMessageSize()}, bufferedAmount=${S.dc.bufferedAmount}`;
    });

    if (!S.role || S.role === "B" || !live()) {
      for (const n of ["real activation transfer: A -> B -> A for each reference case", "Device B rejects a NaN activation and a corrupted packet sent over the real channel"])
        await R.check(G, n, () => skip(!live() ? "no live pipeline" : "run this on Device A (the host): only A starts inference"));
      if (live() && S.role === "B") await R.check(G, "Device B statistics", () => `activations processed ${S.pipe.stats.processed}, rejected ${S.pipe.stats.rejected}, messages received ${S.pipe.stats.received}`);
      return summarise(R.rows, "two-device checks");
    }

    let D;
    try { D = await data(); } catch (e) { await R.check(G, "load reference fixture", () => { throw e; }); return summarise(R.rows, "two-device checks"); }
    const MP = MeshProtocol, TOL = D.fixture.tolerance, cases = pickCases(D.fixture, only);
    for (const c of cases) {
      await R.check(G, `real transfer "${c.name}" (T=${c.T}): response matches, token and logits vs PyTorch`, async () => {
        if (S.running) skip("a normal inference is running in the UI; try again when it is idle");
        const r = await S.pipe.infer(c.ids, { wantLogits: true, timeoutMs: 120000 });
        need(r.seq === 1 && r.sessionId !== "0", "reply not tied to a session/seq");
        need(r.token === c.argmax, `token ${r.token} != PyTorch argmax ${c.argmax}`);
        const d = maxAbs(r.logits, c.logits);
        need(d <= TOL, `logits differ from PyTorch by ${d.toExponential(2)} (> ${TOL})`);
        return `${r.activationBytes} B in ${r.frames} frame(s), logits diff ${d.toExponential(2)}, rtt ${r.rttMs.toFixed(0)} ms (A ${r.computeMsA.toFixed(0)} / B ${(r.computeMsB || 0).toFixed(0)} ms)`;
      });
    }
    await R.check(G, "Device B rejects a NaN activation and a corrupted packet sent over the real channel", async () => {
      const seen = [], prev = S.pipe.onEvent;
      S.pipe.onEvent = (t, i) => { if (t === "peer-error") seen.push(i); if (prev) prev(t, i); };    // watch for ERROR packets coming back from B
      try {
        const x = new Float32Array(2 * 384).fill(0.25), bad = x.slice(), sid = BigInt(Date.now()); bad[3] = NaN;
        await dcTransport.send(MP.encodeActivation({ sessionId: sid, seq: 1, activations: bad, shape: [1, 2, 384] }));
        await until(() => seen.some((e) => e.code === "NON_FINITE"), 8000);
        const pkt = MP.encodeActivation({ sessionId: sid + 1n, seq: 1, activations: x, shape: [1, 2, 384] }); new Uint8Array(pkt)[200] ^= 1;
        await dcTransport.send(pkt);
        await until(() => seen.some((e) => e.code === "CHECKSUM"), 8000);
        return "B answered NON_FINITE and CHECKSUM error packets";
      } finally { S.pipe.onEvent = prev; }
    });
    return summarise(R.rows, "two-device checks");
  }

  async function validate(opts = {}) {
    console.log("[meshStage5] LOCAL checks run in this tab only (in-memory transport, they do NOT test WebRTC). TWO-DEVICE checks need a live connection and are SKIPPED otherwise.");
    const a = await local(opts), b = await twoDevice(opts);
    const total = { pass: a.pass + b.pass, fail: a.fail + b.fail, skip: a.skip + b.skip, local: a, twoDevice: b };
    console.log(`[meshStage5] TOTAL: ${total.pass} passed, ${total.fail} failed, ${total.skip} skipped` + (total.skip ? ". Skipped checks were NOT run." : ""));
    return total;
  }

  window.meshStage5 = { validate, local, twoDevice };
})();
