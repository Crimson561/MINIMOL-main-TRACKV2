"use strict";
/* MINIMOL frontend. Real path: Socket.IO signalling -> WebRTC "mesh" DataChannel -> MeshInference + MeshProtocol.
   CONFIG paths and event names match room-server/server.js; room-server/test/signalling.test.mjs exercises the events. */
const CONFIG = {
  paths: {
    socketClient: "/socket.io/socket.io.js",
    protocol: "/protocol/mesh_protocol.js",
    engine: "/engine/inference_engine.js",
    pipeline: "/engine/mesh_pipeline.js",                 // Stage 5: PipelineA / PipelineB (served by the same /engine route)
    manifest: "/weights/weights_manifest.json",           // from export_weights.py -> room-server/public/weights
    a: "/weights/partition_a.bin", b: "/weights/partition_b.bin",
  },
  events: { create: "create-room", join: "join-room", peerJoined: "peer-joined",       // named in ARCHITECTURE.md
            roomCreated: "room-created", offer: "offer", answer: "answer", ice: "ice-candidate" },
  joinPayload: (code) => ({ roomCode: code }),
  codeFrom: (r) => (typeof r === "string" ? r : r && (r.roomCode || r.code || r.room)),
  stun: "stun:stun.l.google.com:19302", newTokens: 24, timeoutMs: 20000,
  // DataChannel send queue: wait when more than HIGH bytes are buffered, resume at LOW (same values dc-test.js uses).
  bufferHigh: 1048576, bufferLow: 262144,
  // Used ONLY if the browser reports no usable pc.sctp.maxMessageSize: 64 KiB is the interoperable default of RFC 8831.
  fallbackMaxMessage: 65536,
  reassemblyMs: 10000,     // a chunked message that stops arriving is discarded after this much silence
  maxEarlyMessages: 256,   // messages kept while the pipeline does not exist yet (normally just the peer's HELLO)
  // Stage 6 reliability. Every wait below is bounded: nothing in the connection code waits forever or retries forever.
  roomCodeMs: 6000,              // the room-server must answer "create room" within this
  establishMs: 30000,            // first connection: offer/answer/ICE must finish within this
  reconnectEstablishMs: 15000,   // each reconnection attempt
  disconnectGraceMs: 8000,       // pc "disconnected" can heal by itself (network blip): wait this long before calling it lost
  maxReconnects: 2, reconnectBackoffMs: 1000,   // Device A re-offers up to 2 times, waiting 1 s then 2 s
  reconnectWaitMs: 20000,        // Device B waits this long for each new offer from A
};
const $ = (id) => document.getElementById(id);
const S = { socket: null, pc: null, dc: null, role: null, room: null, live: false, demo: false, A: null, B: null,
  load: "Not loaded", bytes: 0, rtts: [], running: false, stage: "idle", lastA: 0, lastB: 0,
  // Stage 5: pipe = PipelineA (Device A) or PipelineB (Device B); fingerprint = model identity sent in HELLO;
  // early = messages received before pipe existed; rx = promise chain so messages are handled strictly in arrival order.
  pipe: null, fingerprint: null, loadError: null, early: [], rx: Promise.resolve(), peerState: "no HELLO yet",
  // Stage 6: gen = "connection generation": bumped every time a peer connection is closed/replaced. Every callback captures the
  // generation it was created in and does nothing if S.gen has moved on (that is how an OLD session cannot touch a NEW one).
  gen: 0, timers: new Set(), iceBuf: [], reconnects: 0, peerGone: false, socketUp: false, failReason: null,
  establishTimer: null, joinTimer: null, roomTimer: null, graceTimer: null };
// Stage 6: the single source of truth for "what is the connection doing" (rules live in connection_state.js; the old flags above are display details).
const conn = MeshConnState.createConnectionMachine({ onChange: (to, from, event) => { note(`State: ${from} → ${to} (${event})`); render(); } });
const log = [];
const ts = () => new Date().toLocaleTimeString([], { hour12: false });
function note(msg) { log.unshift({ t: ts(), msg }); renderLog(); }
function renderLog() { $("log").innerHTML = log.length ? log.map((l) => `<li><time>${l.t}</time><span>${l.msg}</span></li>`).join("") : "<li>No events yet.</li>"; }

/* ---------- navigation ---------- */
function go(screen) {
  if (screen === "workspace" && !S.live && !S.demo) return;
  document.body.dataset.screen = screen;
  ["landing", "matchmaker", "workspace"].forEach((s) => ($(s).hidden = s !== screen));
  scrollTo(0, 0);
  if (screen === "workspace") render();
}
document.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", (e) => { e.preventDefault(); if (b.dataset.go === "landing") reset(); go(b.dataset.go); }));
function view(v) {
  document.querySelectorAll(".view").forEach((s) => (s.hidden = s.id !== "v-" + v));
  document.querySelectorAll(".ws-nav button").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
  render();
}
document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => view(b.dataset.view)));

/* ---------- device info (real, from the browser) ---------- */
const ua = navigator.userAgent;
const me = { name: /iPhone/.test(ua) ? "iPhone" : /iPad/.test(ua) ? "iPad" : /Android/.test(ua) ? "Android device" : /Mac/.test(ua) ? "Mac" : /Windows/.test(ua) ? "Windows PC" : /Linux/.test(ua) ? "Linux device" : "This device",
  type: /Mobi|Android|iPhone/.test(ua) ? "Smartphone" : "Computer" };
$("devName").textContent = me.name; $("devType").textContent = me.type;
$("transportName").textContent = "Socket.IO + WebRTC";

/* ---------- matchmaker states ---------- */
const MSG = { idle: "Create a room or join one to begin.", init: "Initialising…", waiting: "Waiting for another device…", found: "Device found.",
  establishing: "Establishing connection…", connected: "Connection successful.", failed: "Connection failed." };
function stage(st, extra) {
  S.stage = st; $("stage").dataset.state = st; $("mmStatus").textContent = extra || MSG[st];
  const busy = ["init", "waiting", "found", "establishing"].includes(st);
  // The host must keep seeing the room code it has to read out to the other device (it lives inside #panes), so while
  // waiting as host keep the panes visible but show only the code, not the Create/Join controls.
  const hostWaiting = st === "waiting" && S.role === "A" && !!S.room;
  $("cancelBtn").hidden = !busy; $("panes").hidden = (busy && !hostWaiting) || st === "connected";
  $("createBtn").hidden = hostWaiting; $("panes").children[1].hidden = hostWaiting;
  $("matchBox").hidden = st !== "connected"; $("failBox").hidden = st !== "failed";
  if (st === "failed") $("failMsg").textContent = extra || MSG.failed;
}

/* ---------- Stage 6: timers, cleanup, failure ---------- */
// Every timeout goes through later() so that cleanup can cancel ALL of them. A timer nobody can cancel is how an old session's
// "timed out" fires into a brand-new session.
function later(fn, ms) { const id = setTimeout(() => { S.timers.delete(id); fn(); }, ms); S.timers.add(id); return id; }
function cancel(id) { if (id !== null && id !== undefined) { clearTimeout(id); S.timers.delete(id); } }
function clearTimers() { S.timers.forEach((id) => clearTimeout(id)); S.timers.clear(); S.establishTimer = S.joinTimer = S.roomTimer = S.graceTimer = null; }

/** Release the peer connection and everything hanging off it. Idempotent: calling it twice (or with nothing open) is harmless. */
function closePeer(reason) {
  S.gen++;                                     // from now on every callback of the old connection sees gen !== S.gen and returns
  clearTimers();
  dropPipeline(reason);
  const { dc, pc } = S; S.dc = S.pc = null; S.live = false; S.iceBuf = []; S.rx = Promise.resolve();
  if (dc) { dc.onopen = dc.onclose = dc.onclosing = dc.onerror = dc.onmessage = null; try { dc.close(); } catch (e) {} }   // detach first: closing must not call our own handlers
  if (pc) { pc.onicecandidate = pc.onconnectionstatechange = pc.oniceconnectionstatechange = pc.ondatachannel = null; try { pc.close(); } catch (e) {} }
}
/** closePeer + drop the signalling socket. Idempotent. */
function teardown(reason = "connection closed") {
  closePeer(reason);
  const sock = S.socket; S.socket = null; S.socketUp = false;
  if (sock) { try { sock.off(); sock.disconnect(); } catch (e) {} }          // off() first: our own disconnect must not look like a failure
}
function fail(msg) {
  if (conn.is("DISCONNECTED", "FAILED")) return;                               // already idle or already failed: a stale timer must not re-fail
  note("Connection failed: " + msg); S.failReason = msg;
  conn.send("FAIL"); teardown(msg); stage("failed", msg);
  if (document.body.dataset.screen === "workspace") go("matchmaker");        // reuse the existing "Connection failed" box instead of leaving a dead workspace
}
function reset() {
  conn.send("RESET"); teardown("session reset");
  Object.assign(S, { role: null, room: null, A: null, B: null, load: "Not loaded", loadError: null, fingerprint: null, peerState: "no HELLO yet", demo: false, bytes: 0, rtts: [], reconnects: 0, peerGone: false, failReason: null });
  $("codeBox").hidden = true; stage("idle");
}

/* ---------- loading scripts and weights ---------- */
const loadScript = (src) => new Promise((ok, no) => { const s = document.createElement("script"); s.src = src; s.onload = ok; s.onerror = () => no(new Error("Could not load " + src)); document.head.appendChild(s); });
async function libs() {
  if (!window.io) await loadScript(CONFIG.paths.socketClient);
  if (!window.MeshProtocol) await loadScript(CONFIG.paths.protocol);
  if (!window.MeshInference) await loadScript(CONFIG.paths.engine);
  if (!window.MeshPipeline) await loadScript(CONFIG.paths.pipeline);   // must come AFTER protocol + engine: it reads both when it loads
}
async function loadPartition(which) {
  S.loadError = null;
  try {
    S.load = "Loading…"; render();
    const [m, buf] = await Promise.all([
      fetch(CONFIG.paths.manifest).then((r) => { if (!r.ok) throw new Error("weights_manifest.json: HTTP " + r.status); return r.json(); }),
      fetch(CONFIG.paths[which]).then((r) => { if (!r.ok) throw new Error("partition_" + which + ".bin: HTTP " + r.status); return r.arrayBuffer(); })]);
    // Stage 5: refuse the file unless its size and Adler-32 equal what export_weights.py recorded in the manifest.
    // verifyPartitionBuffer throws MeshError("MODEL_MISMATCH") and we deliberately do NOT build an engine in that case.
    const sum = MeshPipeline.verifyPartitionBuffer(buf, m, which);
    S.fingerprint = MeshPipeline.modelFingerprint(m);
    const P = which === "a" ? MeshInference.MeshPartitionA : MeshInference.MeshPartitionB;
    S[which === "a" ? "A" : "B"] = P.fromBuffer(buf, m);
    S.load = `Ready (${(buf.byteLength / 1048576).toFixed(1)} MB, Adler-32 ${sum} verified)`; note("Model partition loaded: " + S.load);
  } catch (e) {
    S[which === "a" ? "A" : "B"] = null; S.fingerprint = null;
    S.loadError = e.code === "MODEL_MISMATCH" ? "Weights REJECTED: " + e.message : "Could not load weights: " + e.message;
    S.load = e.code === "MODEL_MISMATCH" ? "Rejected (checksum/size mismatch)" : "Failed to load weights";
    note(S.loadError); console.error("[MINIMOL] " + S.loadError);
  }
  render(); maybeInitPipeline();   // the DataChannel may already be open
}

/* ---------- signalling + WebRTC ---------- */
async function connect(role) {
  if (!conn.is("DISCONNECTED")) throw new Error("A connection is already in progress.");   // Stage 6: never initialise a second connection on top of a live one
  S.role = role; S.reconnects = 0; S.peerGone = false; S.failReason = null; conn.send("START"); stage("init");
  try { await libs(); } catch (e) { fail(e.message + ". Open this page from the room-server."); throw e; }
  if (!conn.is("CONNECTING")) throw new Error("Cancelled while loading.");                 // the user pressed Cancel during the await
  // reconnection:false: Socket.IO would otherwise silently reconnect with a NEW socket id, and the server has already forgotten our room.
  const sock = io({ reconnection: false }); S.socket = sock;
  await new Promise((ok, no) => { const t = setTimeout(() => no(new Error("Cannot reach the room-server.")), 6000); sock.on("connect", () => { clearTimeout(t); ok(); }); sock.on("connect_error", () => { clearTimeout(t); no(new Error("Cannot reach the room-server.")); }); }).catch((e) => { if (S.socket === sock) fail(e.message); throw e; });   // if the user cancelled meanwhile, this socket is no longer ours: do not fail a NEWER session
  if (S.socket !== sock) throw new Error("Cancelled while connecting.");
  S.socketUp = true;
  const E = CONFIG.events;
  sock.on(E.peerJoined, () => {
    if (S.socket !== sock || S.role !== "A") return;
    if (!conn.is("CONNECTING") || S.pc || S.joinTimer !== null) return note("Ignored a duplicate peer-joined");   // Stage 6: one connection attempt at a time
    S.peerGone = false; note("Device joined"); stage("found");
    S.joinTimer = later(() => { S.joinTimer = null; if (!conn.is("CONNECTING")) return; stage("establishing"); makePC(true); }, 700);
  });
  sock.on(E.offer, async (d) => {
    if (S.socket !== sock || S.role !== "B" || conn.is("DISCONNECTED", "FAILED")) return;
    if (conn.is("CONNECTED", "INFERENCE_READY")) { note("New offer from Device A: the old connection is replaced"); conn.send("LINK_LOST"); }
    makePC(false);                                                  // closes any older peer connection first, then creates the new one
    const gen = S.gen, pc = S.pc;
    try {
      await pc.setRemoteDescription(d.sdp || d); if (gen !== S.gen) return;
      await flushIce(gen); if (gen !== S.gen) return;
      const a = await pc.createAnswer(); if (gen !== S.gen) return;
      await pc.setLocalDescription(a); if (gen !== S.gen) return;
      sock.emit(E.answer, { sdp: pc.localDescription, roomCode: S.room }); stage("establishing");
    } catch (e) { linkLost("Could not answer the offer: " + e.message, gen); }
  });
  sock.on(E.answer, async (d) => {
    const gen = S.gen, pc = S.pc;
    if (S.socket !== sock || S.role !== "A" || !pc) return;
    if (pc.signalingState !== "have-local-offer") return note("Ignored an unexpected answer");    // duplicate/late answer: applying it would throw and kill a working connection
    try { await pc.setRemoteDescription(d.sdp || d); if (gen !== S.gen) return; await flushIce(gen); }
    catch (e) { linkLost("Could not apply the answer: " + e.message, gen); }
  });
  sock.on(E.ice, (d) => {
    const c = d && (d.candidate || d); if (S.socket !== sock || !S.pc || !c) return;
    if (!S.pc.remoteDescription) { S.iceBuf.push(c); return; }       // Stage 6: a candidate can overtake the offer/answer; adding it now would fail, so hold it
    S.pc.addIceCandidate(c).catch((e) => note("ICE candidate rejected: " + e.message));
  });
  sock.on("peer-left", () => {
    if (S.socket !== sock) return;
    S.peerGone = true; note("The other device left the room");
    if (conn.is("CONNECTING", "RECONNECTING")) fail("The other device left the room.");
    else render();                                                   // connected: the data path may still be alive, so only a dead DataChannel/pc ends the session
  });
  sock.on("disconnect", (why) => {
    if (S.socket !== sock) return;
    S.socketUp = false; note("Signalling socket closed (" + why + ")");
    if (conn.is("CONNECTING")) fail("Lost contact with the room-server.");   // before the DataChannel exists, signalling IS the only path
    else render();                                                            // after it exists, inference does not use signalling: keep going (but no reconnection is possible)
  });
}
/** Apply ICE candidates that arrived before the remote description was set. */
async function flushIce(gen) {
  const q = S.iceBuf; S.iceBuf = [];
  for (const c of q) { if (gen !== S.gen || !S.pc) return; try { await S.pc.addIceCandidate(c); } catch (e) { note("ICE candidate rejected: " + e.message); } }
}
function makePC(host) {
  closePeer("starting a new peer connection");     // never two peer connections at once (also bumps S.gen)
  const gen = S.gen, E = CONFIG.events, pc = new RTCPeerConnection({ iceServers: [{ urls: CONFIG.stun }] }); S.pc = pc;
  pc.onicecandidate = (e) => { if (gen === S.gen && e.candidate && S.socket) S.socket.emit(E.ice, { candidate: e.candidate, roomCode: S.room }); };
  pc.oniceconnectionstatechange = () => { if (gen === S.gen) note("ICE: " + pc.iceConnectionState); };
  pc.onconnectionstatechange = () => {
    if (gen !== S.gen) return;                      // an old connection reporting late
    const c = pc.connectionState; note("Peer connection: " + c);
    if (c === "connected") { cancel(S.graceTimer); S.graceTimer = null; }
    else if (c === "disconnected") { cancel(S.graceTimer); S.graceTimer = later(() => linkLost("Peer connection stayed disconnected", gen), S.peerGone ? 0 : CONFIG.disconnectGraceMs); }   // may heal itself, unless signalling already told us the peer left
    else if (c === "failed" || c === "closed") linkLost("Peer connection " + c, gen);
    render();
  };
  if (host) {
    wire(pc.createDataChannel("mesh", { ordered: true }), gen);
    pc.createOffer().then((o) => gen === S.gen && pc.setLocalDescription(o)).then(() => gen === S.gen && S.socket && S.socket.emit(E.offer, { sdp: pc.localDescription, roomCode: S.room }))
      .catch((e) => linkLost("Could not create the offer: " + e.message, gen));
  } else pc.ondatachannel = (e) => { if (gen === S.gen) wire(e.channel, gen); };
  S.establishTimer = later(() => onEstablishTimeout(gen), conn.is("RECONNECTING") ? CONFIG.reconnectEstablishMs : CONFIG.establishMs);
}
function onEstablishTimeout(gen) {
  if (gen !== S.gen) return;
  if (conn.is("CONNECTING")) fail("Timed out while establishing the connection.");
  else if (conn.is("RECONNECTING")) { closePeer("reconnection attempt timed out"); startReconnect("Timed out re-establishing the connection"); }
}
function wire(dc, gen) {
  S.dc = dc; dc.binaryType = "arraybuffer";
  dc.bufferedAmountLowThreshold = CONFIG.bufferLow;   // fires "bufferedamountlow" when the send queue drains below this (see makeTransport)
  dc.onopen = () => {
    if (gen !== S.gen) return;
    cancel(S.establishTimer); S.establishTimer = null; cancel(S.graceTimer); S.graceTimer = null;
    S.live = true; conn.send("DC_OPEN"); note("Peer connection established"); stage("connected"); maybeInitPipeline();   // weights may already be loaded
  };
  dc.onclose = () => { if (gen !== S.gen) return; note("DataChannel closed"); linkLost("DataChannel closed", gen); };
  dc.onclosing = () => { if (gen === S.gen) note("DataChannel closing"); };
  dc.onerror = (e) => { if (gen === S.gen) note("DataChannel error: " + ((e && e.error && e.error.message) || "unknown")); };   // an error is followed by close; recovery happens there
  note("DataChannel " + dc.readyState);
  dc.onmessage = (e) => { if (gen === S.gen) onPacket(e.data); };
}
/** The data path (peer connection or DataChannel) died. Called from several places, so it must be safe to call repeatedly: only the first call per generation acts. */
function linkLost(reason, gen) {
  if (gen !== S.gen) return;                                  // report from an old session: ignore
  if (conn.is("CONNECTING")) return fail(reason);             // never got going: the user presses Retry
  if (conn.is("RECONNECTING")) { closePeer(reason); return startReconnect(reason); }   // this reconnection attempt failed
  if (!conn.is("CONNECTED", "INFERENCE_READY")) return;
  note("Connection lost: " + reason);
  conn.send("LINK_LOST");                                     // state first, so nothing below can observe "INFERENCE_READY" with a dead channel
  closePeer(reason);                                          // rejects the request in flight, drops the pipeline, clears timers and references
  startReconnect(reason);
}
/** Bounded recovery. Device A re-offers over the still-open signalling socket; Device B waits for that offer. After maxReconnects attempts: FAILED. */
function startReconnect(reason) {
  if (S.peerGone) return fail(reason + ". The other device has left the room.");
  if (!S.socket || !S.socketUp) return fail(reason + ". The signalling connection is gone, so a new connection cannot be negotiated.");
  const plan = MeshConnState.reconnectPlan({ attempts: S.reconnects, max: CONFIG.maxReconnects, baseMs: CONFIG.reconnectBackoffMs });
  if (!plan.retry) return fail(`${reason}. Gave up after ${S.reconnects} reconnection attempt(s).`);
  S.reconnects = plan.attempt;
  const msg = `Connection lost. Reconnecting (attempt ${plan.attempt} of ${CONFIG.maxReconnects})…`; note(msg); stage("establishing", msg);
  if (S.role === "A") later(() => { if (conn.is("RECONNECTING") && !S.pc) makePC(true); }, plan.delayMs);
  else S.establishTimer = later(() => { if (conn.is("RECONNECTING") && !S.pc) startReconnect("Device A did not reconnect"); }, CONFIG.reconnectWaitMs);
}
$("createBtn").onclick = async () => {
  try { await connect("A"); } catch (e) { return; }
  const got = (r) => { const c = CONFIG.codeFrom(r); if (!c || S.room || !conn.is("CONNECTING")) return; cancel(S.roomTimer); S.roomTimer = null; S.room = c; $("codeOut").textContent = c; $("codeBox").hidden = false; note("Room created: " + c); stage("waiting"); loadPartition("a"); };
  S.socket.on(CONFIG.events.roomCreated, got); S.socket.emit(CONFIG.events.create, {}, got);
  S.roomTimer = later(() => { if (!S.room) fail("The room-server did not return a room code. Check CONFIG.events in main.js."); }, CONFIG.roomCodeMs);
};
$("joinBtn").onclick = async () => {
  const code = $("codeIn").value.trim().toUpperCase(); if (!code) return $("codeIn").focus();
  try { await connect("B"); } catch (e) { return; }
  S.room = code; S.socket.emit(CONFIG.events.join, CONFIG.joinPayload(code), (r) => { if (r && (r.error || r.ok === false)) fail(r.error || "Room not found."); });
  note("Joining room " + code); stage("waiting", "Joining room…"); loadPartition("b");
  S.socket.on("error", (m) => fail(typeof m === "string" ? m : "Could not join that room."));
};
$("copyBtn").onclick = async () => {   // navigator.clipboard needs a secure context (https/localhost) AND permission; fall back to execCommand when it is missing or refused
  const text = S.room || "", btn = $("copyBtn");
  const legacy = () => { const ta = document.createElement("textarea"); ta.value = text; ta.style.cssText = "position:fixed;opacity:0"; document.body.appendChild(ta); ta.select(); let r = false; try { r = document.execCommand("copy"); } catch (e) {} ta.remove(); return r; };
  let ok = false;
  if (navigator.clipboard && window.isSecureContext) { try { await navigator.clipboard.writeText(text); ok = true; } catch (e) { ok = false; } }
  if (!ok) ok = legacy();
  btn.textContent = ok ? "Copied" : "Copy failed: select the code manually"; setTimeout(() => (btn.textContent = "Copy code"), 1500);
};
$("cancelBtn").onclick = () => { note("Connection cancelled"); reset(); };
$("retryBtn").onclick = () => reset();
$("enterBtn").onclick = () => go("workspace");
$("demoBtn").onclick = () => { S.demo = true; S.live = false; note("Dev demo mode opened (no peer)"); go("workspace"); };
$("leaveBtn").onclick = () => { reset(); go("landing"); };
$("clearLogBtn").onclick = () => { log.length = 0; renderLog(); };

/* ---------- Stage 5: DataChannel transport + MeshPipeline ---------- */
const P = () => window.MeshProtocol;
const meshErr = (code, msg) => new (P().MeshError)(code, msg);

/** The real limit the browsers negotiated for this connection (read at every send; it is only known once SCTP is up). */
function negotiatedMaxMessage() {
  const m = S.pc && S.pc.sctp && S.pc.sctp.maxMessageSize;
  if (m === Infinity) return P().MAX_REASSEMBLED_BYTES;      // peer declared "no limit": stay inside what our own reassembler accepts
  if (Number.isFinite(m) && m >= 256) return Math.floor(m);   // 256 = smallest size splitIntoChunks accepts
  return CONFIG.fallbackMaxMessage;                          // browser reported nothing usable
}
// Resolves on `evt`; rejects if the channel closes first or ms passes (same idea as dc-test.js, kept separate so main.js does not depend on it).
function waitDC(dc, evt, ms) {
  return new Promise((ok, no) => {
    const done = (fn, v) => { clearTimeout(t); dc.removeEventListener(evt, onEvt); dc.removeEventListener("close", onClose); fn(v); };
    const onEvt = () => done(ok), onClose = () => done(no, meshErr("CHANNEL_CLOSED", "DataChannel closed while waiting for " + evt));
    const t = setTimeout(() => done(no, meshErr("SEND_FAILED", "Timed out waiting for DataChannel " + evt)), ms);
    dc.addEventListener(evt, onEvt); dc.addEventListener("close", onClose);
  });
}
/** The object PipelineA/PipelineB expect: { send(ArrayBuffer) -> Promise, maxMessageSize() -> number }. Chunking itself is done by the pipeline.
    Stage 6: a transport is created FOR ONE CONNECTION GENERATION. Once that connection is replaced (S.gen moves on) it refuses to send,
    so a leftover pipeline can never write into the new session's channel. gen === null means "always the current one" (used by the console helpers). */
function makeTransport(gen) {
  const stale = () => gen !== null && gen !== S.gen;
  const gone = () => meshErr("CHANNEL_CLOSED", "this connection has been replaced or closed");
  return {
    maxMessageSize: negotiatedMaxMessage,
    async send(data) {
      if (stale()) throw gone();
      const dc = S.dc;
      if (!dc) throw meshErr("CHANNEL_CLOSED", "no DataChannel");
      if (dc.readyState === "connecting") { await waitDC(dc, "open", 5000); if (stale()) throw gone(); }
      if (dc.readyState !== "open") throw meshErr("CHANNEL_CLOSED", "DataChannel is " + dc.readyState);
      if (data.byteLength > negotiatedMaxMessage()) throw meshErr("SEND_FAILED", `frame of ${data.byteLength} bytes exceeds the DataChannel limit ${negotiatedMaxMessage()}`);
      while (dc.bufferedAmount > CONFIG.bufferHigh) {                       // backpressure: let the browser's send queue drain first
        await waitDC(dc, "bufferedamountlow", CONFIG.timeoutMs);            // rejects CHANNEL_CLOSED if the channel closes while we wait; SEND_FAILED after the timeout
        if (stale()) throw gone();
        if (dc.readyState !== "open") throw meshErr("CHANNEL_CLOSED", "DataChannel is " + dc.readyState);
      }
      try { dc.send(data); } catch (e) { throw meshErr("SEND_FAILED", e.message); }
      S.bytes += data.byteLength;
    },
  };
}
const dcTransport = makeTransport(null);   // kept for Frontend/stage5-test.js, which sends test packets on whatever connection is current

/** Events from the pipeline -> Activity log / metrics. Logging only: nothing here changes pipeline behaviour. */
function pipeEvent(type, i) {
  if (type === "hello") { S.peerState = i.ok ? "verified (same model and weights)" : i.reason; note(i.ok ? "Peer verified: same model and weights" : "Peer HELLO rejected: " + i.reason); }
  else if (type === "activation") note("Activation transferred (" + (i.bytes / 1024).toFixed(1) + " KB)");   // Device B
  else if (type === "result") { S.lastB = i.computeMs; S.reconnects = 0; }                              // Device B served a request: the link really works, so the retry budget is refilled
  else if (type === "error") note(`Pipeline error [${i.code}]: ${i.message}`);
  else if (type === "peer-error") note(`Other device reported error [${i.code}]: ${i.message}`);
  else if (type === "warn") note("Pipeline: " + i.message);
  render();
}

/** Create the pipeline exactly once, and only when BOTH are true: this device's partition is loaded AND the DataChannel is open.
    Called from dc.onopen and from the end of loadPartition(); whichever happens second actually initialises. */
function maybeInitPipeline() {
  if (S.pipe || S.demo || !S.role) return;
  const engine = S.role === "A" ? S.A : S.B;
  if (!engine || !S.fingerprint || !S.dc || S.dc.readyState !== "open") return;
  if (!window.MeshPipeline) return note("Pipeline script (mesh_pipeline.js) is not loaded.");
  const Pipe = S.role === "A" ? MeshPipeline.PipelineA : MeshPipeline.PipelineB;
  S.pipe = new Pipe({ engine, fingerprint: S.fingerprint, transport: makeTransport(S.gen), timeoutMs: CONFIG.timeoutMs, reassemblyTimeoutMs: CONFIG.reassemblyMs, onEvent: pipeEvent });
  conn.send("PIPELINE_READY");
  note(`Pipeline ${S.role} ready. DataChannel message limit: ${negotiatedMaxMessage()} bytes`);
  const queued = S.early; S.early = []; queued.forEach(deliver);           // anything that arrived before we were ready (e.g. the peer's HELLO)
  S.pipe.sendHello().catch((e) => note("Could not send HELLO: " + e.message));   // PipelineBase never replies to HELLO, so each side sends its own
  render();
}
function dropPipeline(reason) {
  const pipe = S.pipe; S.pipe = null; S.early = []; S.peerState = "no HELLO yet";
  if (pipe) { conn.send("PIPELINE_DROPPED"); pipe.dispose(reason); }   // Stage 6: public, idempotent: rejects the request in flight, waiters and timers at once
}

/* ---------- inference ---------- */
// One entry point for every incoming DataChannel message. dc-test.js traffic is claimed first; everything else is MeshPipeline's.
function onPacket(raw) {
  if (window.meshTest && window.meshTest.handle(raw)) return;   // Stage 3/4: DataChannel test traffic (see dc-test.js)
  if (!(raw instanceof ArrayBuffer)) return note("Ignored a non-binary message");
  S.bytes += raw.byteLength;
  if (!S.pipe) {                                                // not ready yet: keep it, maybeInitPipeline() replays it
    if (S.early.length < CONFIG.maxEarlyMessages) S.early.push(raw); else note("Dropped a message: pipeline not ready and queue is full");
    return;
  }
  deliver(raw);
}
// Strictly one message at a time and in arrival order (chunk frames of one packet must not overtake each other). handleMessage never throws.
function deliver(raw) { const pipe = S.pipe; S.rx = S.rx.then(() => pipe.handleMessage(raw)).catch((e) => note("Pipeline failure: " + e.message)); }

async function runLive(prompt, out) {
  if (S.role !== "A") throw new Error("Start inference from Device A (the host).");
  if (S.loadError) throw new Error(S.loadError);
  if (!S.A) throw new Error("Partition A is not loaded.");
  if (!conn.is("INFERENCE_READY") || !S.pipe) throw new Error(conn.is("RECONNECTING") ? "Connection lost: reconnecting. Try again in a moment." : conn.is("CONNECTED") ? "Pipeline is not ready yet: it needs the loaded model partition and an open DataChannel." : "Not connected to Device B.");
  const tokens = Array.from(new TextEncoder().encode(prompt)), bytes = [], max = MeshInference.MAX_SEQ_LEN;
  if (tokens.length > max) throw new Error(`Prompt is ${tokens.length} bytes; the model context is ${max}.`);
  note("Inference started");
  for (let i = 0; i < CONFIG.newTokens && tokens.length < max; i++) {
    const r = await S.pipe.infer(tokens);            // A: layers 0-2 -> activation -> B: layers 3-5 + head -> next byte (session/seq matched inside)
    S.lastA = r.computeMsA; S.lastB = r.computeMsB || 0; S.rtts.push(r.rttMs); S.reconnects = 0;   // a full A->B->A round trip: the link is proven good
    tokens.push(r.token); bytes.push(r.token);
    out(new TextDecoder().decode(new Uint8Array(bytes)), i + 1);
  }
  note("Inference completed");
}
async function runDemo(prompt, out) {   // DEMO ONLY: canned text, not model output
  const txt = "[demo text, not generated by the model] "; for (let i = 1; i <= txt.length; i++) { await new Promise((r) => setTimeout(r, 40)); out(txt.slice(0, i), i); }
}
$("runBtn").onclick = async () => {
  const prompt = $("prompt").value.trim(); if (!prompt || S.running) return;
  S.running = true; $("runBtn").disabled = true; $("genStatus").textContent = "Generating…"; $("out").textContent = ""; $("mA").textContent = $("mB").textContent = "working";
  const t0 = performance.now();
  const out = (text, n) => { $("out").textContent = text; const d = (performance.now() - t0) / 1000; $("mTok").textContent = n; $("mDur").textContent = d.toFixed(1) + "s"; $("mTps").textContent = (n / d).toFixed(1);
    if (!S.demo) { $("mA").textContent = S.lastA.toFixed(0) + " ms"; $("mB").textContent = S.lastB.toFixed(0) + " ms"; } };
  try { S.demo ? await runDemo(prompt, out) : await runLive(prompt, out); $("genStatus").textContent = "Done"; }   // Stage 6: a live session that lost its link must say so, never fall back to canned demo text
  catch (e) { $("genStatus").textContent = e.message; note("Inference error: " + e.message); }
  S.running = false; $("runBtn").disabled = false; if (S.demo) $("mA").textContent = $("mB").textContent = "demo"; render();
};
$("clearBtn").onclick = () => { $("out").textContent = ""; $("prompt").value = ""; ["mTok", "mTps"].forEach((i) => ($(i).textContent = "0")); $("mDur").textContent = "0.0s"; $("genStatus").textContent = "Idle"; };

/* ---------- workspace rendering ---------- */
const flow = (live) => `<div class="${live ? "live" : ""}"><b>DEVICE A</b>Embedding + Transformer layers 0–2</div><div class="arrow">↓</div><div class="xfer"><b>ACTIVATION TRANSFER</b>float32 [1, T, 384] over the WebRTC DataChannel</div><div class="arrow">↓</div><div class="${live ? "live" : ""}"><b>DEVICE B</b>Transformer layers 3–5 + final LayerNorm + language model head</div><div class="arrow">↓</div><div><b>GENERATED TOKEN</b>one byte (0–255) returned to Device A</div>`;
$("flowLanding").innerHTML = flow(false);
function render() {
  if (document.body.dataset.screen !== "workspace") return;
  const open = S.dc && S.dc.readyState === "open";
  const mode = S.demo ? "DEV DEMO" : conn.is("RECONNECTING") ? "Reconnecting…" : conn.is("FAILED") ? "Failed" : open && conn.is("INFERENCE_READY") ? "Live" : open ? "Connected (preparing)" : "Disconnected";
  $("banner").className = "banner " + (S.demo ? "demo" : "live");
  $("banner").textContent = S.demo ? "DEV DEMO MODE. No peer is connected. Connection state, devices and inference output here are not real." : `Live session. Room ${S.room || "—"}. This device is Device ${S.role}.`;
  const rtt = S.rtts.length ? (S.rtts.reduce((a, b) => a + b) / S.rtts.length).toFixed(0) + " ms" : "—";
  const mine = S.role === "A" ? "A" : "B";
  $("ovA").innerHTML = `<span class="dot ${open && !S.demo ? "ok" : ""}"></span>` + (S.demo ? "Demo" : mine === "A" ? me.name + " (this device)" : "Peer device");
  $("ovB").innerHTML = `<span class="dot ${open && !S.demo ? "ok" : ""}"></span>` + (S.demo ? "Demo" : mine === "B" ? me.name + " (this device)" : "Peer device");
  $("ovAs").textContent = "Embedding + layers 0–2"; $("ovBs").textContent = "Layers 3–5 + head";
  $("ovConn").textContent = mode; $("ovTrans").textContent = S.demo ? "No transport" : "WebRTC DataChannel \"mesh\"";
  $("ovModel").textContent = "Byte Micro-Transformer"; $("ovLoad").textContent = "Partition " + mine + ": " + (S.demo ? "not loaded (demo)" : S.load);
  $("ovInf").textContent = S.running ? "Running" : "Idle"; $("ovInfS").textContent = S.live ? `${(S.bytes / 1024).toFixed(1)} KB transferred` : "";
  $("model").innerHTML = "<option>Byte Micro-Transformer, 11.04M parameters</option>";
  $("engineTag").textContent = S.demo ? "Demo output" : "Live engine"; $("flowWs").innerHTML = flow(S.running);
  $("runBtn").title = !S.demo && S.role === "B" ? "Inference is started from Device A" : "";
  const card = (n, role, parts, here) => `<div class="card"><h3>${n}${here ? " (this device)" : ""}</h3><dl><dt>Type</dt><dd>${here ? me.type : S.demo ? "—" : "Remote device"}</dd><dt>Status</dt><dd>${S.demo ? "Demo" : open ? "Connected" : "Disconnected"}</dd><dt>Runs</dt><dd>${parts}</dd><dt>Latency</dt><dd>${rtt === "—" ? "Measured during inference" : "Round trip " + rtt}</dd><dt>Data</dt><dd>${S.live ? (S.bytes / 1024).toFixed(1) + " KB" : "—"}</dd></dl></div>`;
  $("devCards").innerHTML = card("Device A", "", "Embedding + layers 0–2", !S.demo && mine === "A") + card("Device B", "", "Layers 3–5 + LayerNorm + head", !S.demo && mine === "B");
  $("setInfo").textContent = S.demo ? "Dev demo mode." : `Room ${S.room}. Role: Device ${S.role}. Signalling: Socket.IO. Weights: ${S.load}. Pipeline: ${S.pipe ? "ready, peer " + S.peerState : "not initialised"}.`;
}
renderLog(); stage("idle"); view("overview"); setInterval(render, 2000);
