import path from "path";
import { fileURLToPath } from "url";
import express from "express";
import { createServer } from "http";
import { Server } from "socket.io";
import { randomInt } from "crypto";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express()
app.use(express.static(path.join(__dirname, "../Frontend")));
app.use("/protocol", express.static(path.join(__dirname, "../protocol")));
app.use("/engine", express.static(path.join(__dirname, "../engine")));
app.use("/weights", express.static(path.join(__dirname, "public/weights")));
app.use("/fixtures", express.static(path.join(__dirname, "../tests/fixtures")));   // Stage 5: PyTorch reference data for the browser check
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "../Frontend/main.html"));
});
const port = 3000
const httpServer = createServer(app)
const io = new Server(httpServer);

/* ---------- room matchmaking state ---------- */
const MAX_PEERS = 2;                                    // one room = exactly two devices (A and B)
const CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"; // no 0/O/1/I so codes are easy to read aloud
const rooms = new Map();                                // roomCode -> Set of socket ids currently inside

function generateRoomCode() {
  let code;
  do {
    code = "MESH-";
    for (let i = 0; i < 5; i++) code += CODE_ALPHABET[randomInt(CODE_ALPHABET.length)];
  } while (rooms.has(code));                            // regenerate on the (unlikely) collision
  return code;
}
const respond = (ack, payload) => { if (typeof ack === "function") ack(payload); };
const refuse = (ack, code, error) => respond(ack, { ok: false, code, error });


io.on("connection", (socket) => {
  console.log(`client connect hogya oyeeeeeeeeee: ${socket.id}`);

socket.on("ping-server", (data) => {
  console.log(`ping server recieve hogya!! `, data);

    socket.broadcast.emit("server-reply" , {
    message: data , 
    timestamp: Date.now()
  });
});
  socket.on("create-room", (_data, ack) => {
    if (socket.data.roomCode) return refuse(ack, "ALREADY_IN_ROOM", "This device is already in a room.");
    const roomCode = generateRoomCode();
    rooms.set(roomCode, new Set([socket.id]));
    socket.join(roomCode);
    socket.data.roomCode = roomCode;
    console.log(`${socket.id} created room ${roomCode}`);
    socket.emit("room-created", { roomCode });          // the frontend listens for this event...
    respond(ack, { ok: true, roomCode });               // ...and also accepts it as the emit() acknowledgement
  });

  // Accepts { roomCode } (what the frontend sends) or a bare string (what the old test client sent).
  socket.on("join-room" , (payload, ack) => {
    const raw = typeof payload === "string" ? payload : payload && payload.roomCode;
    const roomCode = typeof raw === "string" ? raw.trim().toUpperCase() : "";
    const refuseJoin = (code, error) => {
      console.log(`${socket.id} join refused (${code}): ${roomCode || "<none>"}`);
      if (typeof ack === "function") refuse(ack, code, error);
      else socket.emit("error", error);                 // no ack callback (old test client): fall back to an event
    };
    if (!/^MESH-[A-Z0-9]{5}$/.test(roomCode)) return refuseJoin("INVALID_ROOM_CODE", "That is not a valid room code.");
    const room = rooms.get(roomCode);
    if (!room) return refuseJoin("ROOM_NOT_FOUND", "Room not found.");
    if (room.has(socket.id)) return respond(ack, { ok: true, roomCode, alreadyJoined: true }); // duplicate join: harmless no-op
    if (socket.data.roomCode) return refuseJoin("ALREADY_IN_ROOM", "This device is already in a room.");
    if (room.size >= MAX_PEERS) return refuseJoin("ROOM_FULL", "That room already has two devices.");
    room.add(socket.id);
    socket.join(roomCode);
    socket.data.roomCode = roomCode;
    console.log(`${socket.id} joined room ${roomCode}`);
    socket.to(roomCode).emit("peer-joined", { peerId: socket.id }); // everyone already inside (Device A) is told
    respond(ack, { ok: true, roomCode });
  });

  // Relay a WebRTC signalling message to the OTHER device in the sender's room.
  // The room comes from server-side state, never from the client's own roomCode field.
  const relaySignal = (event, key, isValid) => {
    socket.on(event, (data) => {
      const roomCode = socket.data.roomCode;
      if (!roomCode || !rooms.has(roomCode)) return console.log(`${socket.id} sent ${event} without being in a room`);
      if (!data || !isValid(data[key])) return console.log(`${socket.id} sent malformed ${event}`);
      socket.to(roomCode).emit(event, { [key]: data[key] });
    });
  };
  relaySignal("offer", "sdp", (v) => v && v.type === "offer" && typeof v.sdp === "string");
  relaySignal("answer", "sdp", (v) => v && v.type === "answer" && typeof v.sdp === "string");
  relaySignal("ice-candidate", "candidate", (v) => v && typeof v === "object" && typeof v.candidate === "string");

  socket.on("disconnect", () => {
    const roomCode = socket.data.roomCode;
    const room = roomCode && rooms.get(roomCode);
    if (!room) return;
    room.delete(socket.id);
    console.log(`${socket.id} left room ${roomCode}`);
    // Stage 6: a room is one two-device session. When a member leaves, tell whoever is still inside, then close the room
    // (forget the code, release the remaining sockets) so a stale half-empty room cannot be joined by a third device.
    if (room.size > 0) {
      io.to(roomCode).emit("peer-left", { peerId: socket.id });
      for (const id of room) {
        const other = io.sockets.sockets.get(id);
        if (other) { other.leave(roomCode); other.data.roomCode = undefined; }
      }
    }
    rooms.delete(roomCode);
  });

});


httpServer.listen(port , "0.0.0.0",  () => {
    console.log(`trying to make this express server work on ${port}`)
});


