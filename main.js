"use strict";
/* MINIMOL frontend. Real path: Socket.IO signalling -> WebRTC "mesh" DataChannel -> MeshInference + MeshProtocol.
   CONFIG values marked VERIFY are NOT in the repo zip (room-server was missing): check them against room-server code. */
const CONFIG = {
  paths: {
    socketClient: "/socket.io/socket.io.js",              // VERIFY (default Socket.IO path)
    protocol: "/protocol/mesh_protocol.js",               // VERIFY how room-server serves protocol/
    engine: "/engine/inference_engine.js",                // VERIFY how room-server serves engine/
    manifest: "/weights/weights_manifest.json",           // from export_weights.py -> room-server/public/weights
    a: "/weights/partition_a.bin", b: "/weights/partition_b.bin",
  },
  events: { create: "create-room", join: "join-room", peerJoined: "peer-joined",       // named in ARCHITECTURE.md
            roomCreated: "room-created", offer: "offer", answer: "answer", ice: "ice-candidate" }, // VERIFY these four
  joinPayload: (code) => ({ roomCode: code }),            // VERIFY
  codeFrom: (r) => (typeof r === "string" ? r : r && (r.roomCode || r.code || r.room)), // VERIFY
  stun: "stun:stun.l.google.com:19302", newTokens: 24, timeoutMs: 20000,
};
const $ = (id) => document.getElementById(id);
const S = { socket: null, pc: null, dc: null, role: null, room: null, live: false, demo: false, A: null, B: null,
  load: "Not loaded", seq: 0, pend: null, bytes: 0, rtts: [], running: false, stage: "idle", lastA: 0, lastB: 0 };
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
  $("cancelBtn").hidden = !busy; $("panes").hidden = busy || st === "connected";
  $("matchBox").hidden = st !== "connected"; $("failBox").hidden = st !== "failed";
  if (st === "failed") $("failMsg").textContent = extra || MSG.failed;
}
function fail(msg) { note("Connection failed: " + msg); teardown(); stage("failed", msg); }
function teardown() { try { S.dc && S.dc.close(); S.pc && S.pc.close(); S.socket && S.socket.disconnect(); } catch (e) {} S.dc = S.pc = S.socket = null; S.live = false; }
function reset() { teardown(); Object.assign(S, { role: null, room: null, A: null, B: null, load: "Not loaded", demo: false, bytes: 0, rtts: [] }); $("codeBox").hidden = true; stage("idle"); }

/* ---------- loading scripts and weights ---------- */
const loadScript = (src) => new Promise((ok, no) => { const s = document.createElement("script"); s.src = src; s.onload = ok; s.onerror = () => no(new Error("Could not load " + src)); document.head.appendChild(s); });
async function libs() {
  if (!window.io) await loadScript(CONFIG.paths.socketClient);
  if (!window.MeshProtocol) await loadScript(CONFIG.paths.protocol);
  if (!window.MeshInference) await loadScript(CONFIG.paths.engine);
}
async function loadPartition(which) {
  try {
    S.load = "Loading…"; render();
    const [m, buf] = await Promise.all([fetch(CONFIG.paths.manifest).then((r) => r.json()), fetch(CONFIG.paths[which]).then((r) => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); })]);
    const P = which === "a" ? MeshInference.MeshPartitionA : MeshInference.MeshPartitionB;
    S[which === "a" ? "A" : "B"] = P.fromBuffer(buf, m);
    S.load = `Ready (${(buf.byteLength / 1048576).toFixed(1)} MB)`; note("Model partition loaded: " + S.load);
  } catch (e) { S.load = "Failed to load weights"; note("Model load failed: " + e.message); }
  render();
}

/* ---------- signalling + WebRTC ---------- */
async function connect(role) {
  S.role = role; stage("init");
  try { await libs(); } catch (e) { return fail(e.message + ". Open this page from the room-server."); }
  S.socket = io();
  await new Promise((ok, no) => { const t = setTimeout(() => no(new Error("Cannot reach the room-server.")), 6000); S.socket.on("connect", () => { clearTimeout(t); ok(); }); S.socket.on("connect_error", () => { clearTimeout(t); no(new Error("Cannot reach the room-server.")); }); }).catch((e) => { fail(e.message); throw e; });
  const E = CONFIG.events;
  S.socket.on(E.peerJoined, () => { if (S.role !== "A") return; note("Device joined"); stage("found"); setTimeout(() => { stage("establishing"); makePC(true); }, 700); });
  S.socket.on(E.offer, async (d) => { if (!S.pc) makePC(false); await S.pc.setRemoteDescription(d.sdp || d); const a = await S.pc.createAnswer(); await S.pc.setLocalDescription(a); S.socket.emit(E.answer, { sdp: S.pc.localDescription, roomCode: S.room }); stage("establishing"); });
  S.socket.on(E.answer, (d) => S.pc && S.pc.setRemoteDescription(d.sdp || d));
  S.socket.on(E.ice, (d) => S.pc && (d.candidate || d) && S.pc.addIceCandidate(d.candidate || d).catch(() => {}));
  S.socket.on("disconnect", () => note("Signalling socket closed"));
}
function makePC(host) {
  const E = CONFIG.events; S.pc = new RTCPeerConnection({ iceServers: [{ urls: CONFIG.stun }] });
  S.pc.onicecandidate = (e) => e.candidate && S.socket && S.socket.emit(E.ice, { candidate: e.candidate, roomCode: S.room });
  S.pc.onconnectionstatechange = () => { const c = S.pc && S.pc.connectionState; if (c === "failed") fail("The two devices could not reach each other."); if (c === "disconnected") { note("Connection interrupted"); render(); } };
  if (host) { wire(S.pc.createDataChannel("mesh", { ordered: true })); S.pc.createOffer().then((o) => S.pc.setLocalDescription(o)).then(() => S.socket.emit(E.offer, { sdp: S.pc.localDescription, roomCode: S.room })); }
  else S.pc.ondatachannel = (e) => wire(e.channel);
  setTimeout(() => { if (S.stage === "establishing") fail("Timed out while establishing the connection."); }, CONFIG.timeoutMs + 10000);
}
function wire(dc) {
  S.dc = dc; dc.binaryType = "arraybuffer";
  dc.onopen = () => { S.live = true; note("Peer connection established"); stage("connected"); };
  dc.onclose = () => { note("Connection interrupted"); render(); };
  dc.onmessage = (e) => onPacket(e.data);
}
$("createBtn").onclick = async () => {
  try { await connect("A"); } catch (e) { return; }
  const got = (r) => { const c = CONFIG.codeFrom(r); if (!c || S.room) return; S.room = c; $("codeOut").textContent = c; $("codeBox").hidden = false; note("Room created: " + c); stage("waiting"); loadPartition("a"); };
  S.socket.on(CONFIG.events.roomCreated, got); S.socket.emit(CONFIG.events.create, {}, got);
  setTimeout(() => { if (!S.room) fail("The room-server did not return a room code. Check CONFIG.events in main.js."); }, 6000);
};
$("joinBtn").onclick = async () => {
  const code = $("codeIn").value.trim().toUpperCase(); if (!code) return $("codeIn").focus();
  try { await connect("B"); } catch (e) { return; }
  S.room = code; S.socket.emit(CONFIG.events.join, CONFIG.joinPayload(code), (r) => { if (r && (r.error || r.ok === false)) fail(r.error || "Room not found."); });
  note("Joining room " + code); stage("waiting", "Joining room…"); loadPartition("b");
  S.socket.on("error", (m) => fail(typeof m === "string" ? m : "Could not join that room."));
};
$("copyBtn").onclick = () => navigator.clipboard.writeText(S.room || "").then(() => { $("copyBtn").textContent = "Copied"; setTimeout(() => ($("copyBtn").textContent = "Copy code"), 1500); });
$("cancelBtn").onclick = () => { note("Connection cancelled"); reset(); };
$("retryBtn").onclick = () => reset();
$("enterBtn").onclick = () => go("workspace");
$("demoBtn").onclick = () => { S.demo = true; S.live = false; note("Dev demo mode opened (no peer)"); go("workspace"); };
$("leaveBtn").onclick = () => { reset(); go("landing"); };
$("clearLogBtn").onclick = () => { log.length = 0; renderLog(); };

/* ---------- inference ---------- */
const P = () => window.MeshProtocol;
function onPacket(raw) {
  let p; try { p = P().decodePacket(raw); } catch (e) { return note("Rejected packet: " + e.message); }
  S.bytes += raw.byteLength;
  if (p.msgType === P().MSG_TYPES.ACTIVATION && S.role === "B") {            // Device B: layers 3-5 + head
    if (!S.B) return dcSend(P().encodeControl({ sessionId: BigInt(p.sessionId), command: "ERROR", data: { reason: "partition B not loaded" } }));
    note("Activation transferred (" + (raw.byteLength / 1024).toFixed(1) + " KB)"); render();
    const t0 = performance.now(); const r = S.B.forward(p.payload, p.shape[1]); const ms = performance.now() - t0; S.lastB = ms;
    dcSend(P().encodeToken({ sessionId: BigInt(p.sessionId), seq: p.seq, token: r.nextByte, timing: { computeMs: ms } })); render();
  } else if (p.msgType === P().MSG_TYPES.TOKEN && S.pend) { const f = S.pend; S.pend = null; f(p.payload); }
}
function dcSend(data) { if (S.dc && S.dc.readyState === "open") { S.bytes += data.byteLength; S.dc.send(data); } }
async function runLive(prompt, out) {
  if (S.role !== "A") throw new Error("Start inference from Device A (the host).");
  if (!S.A) throw new Error("Partition A is not loaded.");
  const tokens = Array.from(new TextEncoder().encode(prompt)), sid = BigInt(Date.now()), bytes = [];
  note("Inference started");
  for (let i = 0; i < CONFIG.newTokens && tokens.length < 512; i++) {
    let t0 = performance.now(); const act = S.A.forward(tokens); S.lastA = performance.now() - t0;
    t0 = performance.now();
    const reply = await new Promise((ok, no) => { S.pend = ok; dcSend(P().encodeActivation({ sessionId: sid, seq: S.seq++, activations: act, shape: [1, tokens.length, 384] })); setTimeout(() => { if (S.pend === ok) { S.pend = null; no(new Error("Device B did not answer.")); } }, CONFIG.timeoutMs); });
    S.rtts.push(performance.now() - t0); tokens.push(reply.token); bytes.push(reply.token);
    if (reply.timing) S.lastB = reply.timing.computeMs;
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
  try { S.live ? await runLive(prompt, out) : await runDemo(prompt, out); $("genStatus").textContent = "Done"; }
  catch (e) { $("genStatus").textContent = e.message; note("Inference error: " + e.message); }
  S.running = false; $("runBtn").disabled = false; if (S.demo) $("mA").textContent = $("mB").textContent = "demo"; render();
};
$("clearBtn").onclick = () => { $("out").textContent = ""; $("prompt").value = ""; ["mTok", "mTps"].forEach((i) => ($(i).textContent = "0")); $("mDur").textContent = "0.0s"; $("genStatus").textContent = "Idle"; };

/* ---------- workspace rendering ---------- */
const flow = (live) => `<div class="${live ? "live" : ""}"><b>DEVICE A</b>Embedding + Transformer layers 0–2</div><div class="arrow">↓</div><div class="xfer"><b>ACTIVATION TRANSFER</b>float32 [1, T, 384] over the WebRTC DataChannel</div><div class="arrow">↓</div><div class="${live ? "live" : ""}"><b>DEVICE B</b>Transformer layers 3–5 + final LayerNorm + language model head</div><div class="arrow">↓</div><div><b>GENERATED TOKEN</b>one byte (0–255) returned to Device A</div>`;
$("flowLanding").innerHTML = flow(false);
function render() {
  if (document.body.dataset.screen !== "workspace") return;
  const open = S.dc && S.dc.readyState === "open", mode = S.demo ? "DEV DEMO" : open ? "Live" : "Disconnected";
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
  $("setInfo").textContent = S.demo ? "Dev demo mode." : `Room ${S.room}. Role: Device ${S.role}. Signalling: Socket.IO. Weights: ${S.load}.`;
}
renderLog(); stage("idle"); view("overview"); setInterval(render, 2000);
