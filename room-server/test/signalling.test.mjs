// Signalling-server tests: room matchmaking and WebRTC signal relay (Socket.IO).
// Starts its own server on a spare port, so it does not matter whether port 3000 is in use.
//   cd room-server && npm install && npm test
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { io } from "socket.io-client";

const PORT = 3100 + Math.floor(Math.random() * 800);
const here = path.dirname(fileURLToPath(import.meta.url));
const server = spawn(process.execPath, [path.join(here, "..", "server.js")], { env: { ...process.env, PORT: String(PORT) }, stdio: ["ignore", "pipe", "pipe"] });

let passed = 0, failed = 0;
const check = (name, ok, detail = "") => { ok ? passed++ : failed++; console.log(`  ${ok ? "✓" : "✗ FAIL"} ${name}${ok ? "" : " " + detail}`); };
const sockets = [];
const connect = () => new Promise((res, rej) => {
  const s = io(`http://localhost:${PORT}`, { reconnection: false, transports: ["websocket"] });
  sockets.push(s); s.on("connect", () => res(s)); s.on("connect_error", rej);
});
const ask = (s, ev, data) => new Promise((res) => s.emit(ev, data, res));
const next = (s, ev, ms = 1000) => new Promise((res) => { const t = setTimeout(() => res("TIMEOUT"), ms); s.once(ev, (d) => { clearTimeout(t); res(d); }); });

async function waitForServer() {
  await new Promise((res, rej) => {
    const t = setTimeout(() => rej(new Error("server did not start")), 8000);
    server.stdout.on("data", (d) => { if (String(d).includes("listening")) { clearTimeout(t); res(); } });
    server.on("exit", (c) => rej(new Error("server exited early with code " + c)));
  });
}

async function run() {
  await waitForServer();
  console.log(`Signalling server tests (server on port ${PORT})`);
  const A = await connect(), B = await connect(), C = await connect();

  const created = await ask(A, "create-room", {});
  check("create-room returns a MESH-XXXXX code (no 0/O/1/I)", created.ok && /^MESH-[A-HJ-NP-Z2-9]{5}$/.test(created.roomCode), JSON.stringify(created));
  check("a device already in a room cannot create another", (await ask(A, "create-room", {})).code === "ALREADY_IN_ROOM");
  check("malformed code is refused", (await ask(B, "join-room", { roomCode: "nope" })).code === "INVALID_ROOM_CODE");
  check("unknown room is refused", (await ask(B, "join-room", { roomCode: "MESH-AAAAA" })).code === "ROOM_NOT_FOUND");

  const joined = next(A, "peer-joined");
  const j = await ask(B, "join-room", { roomCode: created.roomCode.toLowerCase() });
  check("join succeeds (code is case-insensitive)", j.ok === true, JSON.stringify(j));
  check("host receives peer-joined with the joiner's id", (await joined).peerId === B.id);
  check("a third device is refused (ROOM_FULL)", (await ask(C, "join-room", { roomCode: created.roomCode })).code === "ROOM_FULL");

  const offer = next(B, "offer"); A.emit("offer", { sdp: { type: "offer", sdp: "v=0" } });
  check("offer is relayed A -> B", (await offer).sdp?.type === "offer");
  const answer = next(A, "answer"); B.emit("answer", { sdp: { type: "answer", sdp: "v=0" } });
  check("answer is relayed B -> A", (await answer).sdp?.type === "answer");
  const ice = next(A, "ice-candidate"); B.emit("ice-candidate", { candidate: { candidate: "candidate:1 1 udp 1 1.1.1.1 1 typ host" } });
  check("ICE candidate is relayed B -> A", (await ice).candidate?.candidate?.startsWith("candidate:"));

  const bad = next(B, "offer", 400); A.emit("offer", { sdp: { type: "answer", sdp: "x" } });
  check("an offer whose SDP type is wrong is NOT relayed", (await bad) === "TIMEOUT");
  const stray = next(A, "offer", 400); C.emit("offer", { sdp: { type: "offer", sdp: "v=0" } });
  check("a signal from a device that is in no room is NOT relayed", (await stray) === "TIMEOUT");

  const bId = B.id, left = next(A, "peer-left"); B.disconnect();
  check("peer-left reaches the remaining device with the leaver's id", (await left).peerId === bId);
  check("the room is closed after a member leaves (a late joiner gets ROOM_NOT_FOUND)", (await ask(C, "join-room", { roomCode: created.roomCode })).code === "ROOM_NOT_FOUND");
  check("the remaining device can create a new room", (await ask(A, "create-room", {})).ok === true);
}

try { await run(); } catch (e) { failed++; console.error("  ✗ test run aborted:", e.message); }
sockets.forEach((s) => s.close()); server.kill();
console.log(failed ? `\nSIGNALLING TESTS FAILED: ${passed} passed, ${failed} failed` : `\nSIGNALLING TESTS PASSED: ${passed} passed, 0 failed`);
process.exit(failed ? 1 : 0);
