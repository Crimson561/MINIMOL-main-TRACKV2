/**
 * MINIMOL Binary Protocol for WebRTC DataChannel (wire magic stays "MESH" = 0x4D455348)
 * 
 * Frame Format (48-byte header + binary payload):
 * 0..3   : Magic "MESH" (0x4D, 0x45, 0x53, 0x48)
 * 4      : Protocol Version (0x01)
 * 5      : Message Type:
 *            0x01 = ACTIVATION (Phone A -> Phone B)
 *            0x02 = TOKEN (Phone B -> Phone A)
 *            0x03 = CONTROL (Handshake / Start / Done)
 *            0x04 = ERROR
 *            0x05 = TENSOR (generic tensor, either direction; Stage 4)
 * 6..7   : Flags (uint16, big-endian)
 *            bit 0: is_eos (generation finished)
 * 8..15  : Session ID (uint64, 8 bytes)
 * 16..19 : Sequence Number (uint32)
 * 20     : Data Type (0x01=FP32, 0x02=UINT8, 0x03=INT32, 0x04=UTF8_JSON)
 * 21     : Tensor Rank (uint8, e.g. 3 for [B, T, C])
 * 22..23 : Reserved (uint16)
 * 24..39 : Shape (4 x uint32: [dim0, dim1, dim2, dim3])
 * 40..43 : Payload Byte Length (uint32)
 * 44..47 : Adler-32 Checksum (uint32)
 * 48..N  : Raw Binary Payload
 *
 * Byte order: header fields are big-endian (network order). The tensor payload is
 * raw little-endian float32 (what every browser/Node target uses natively).
 * Tensors carry at most MAX_RANK (4) dimensions; unused shape slots are zero.
 */

"use strict";

const MAGIC = 0x4d455348; // "MESH"
const PROTOCOL_VERSION = 0x01;

const MSG_TYPES = {
  ACTIVATION: 0x01,
  TOKEN: 0x02,
  CONTROL: 0x03,
  ERROR: 0x04,
  TENSOR: 0x05,
};

const DATA_TYPES = {
  FP32: 0x01,
  UINT8: 0x02,
  INT32: 0x03,
  UTF8_JSON: 0x04,
};

const HEADER_SIZE = 48;

/* ---- Stage 4: tensor serialization constants ---- */
const MAX_RANK = 4; // the header has room for exactly 4 shape slots (bytes 24..39)

// Only dtypes the inference engine really uses as tensors. The engine is Float32Array
// everywhere (weights, activations, logits). UINT8/INT32 codes stay reserved, not supported.
const TENSOR_DTYPES = {
  [DATA_TYPES.FP32]: { name: "float32", bytes: 4, ctor: Float32Array },
};

// The payload is raw typed-array memory, so it is only portable between little-endian hosts.
const IS_LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/**
 * Fast Adler-32 checksum calculation
 */
function adler32(buf, offset, length) {
  let a = 1, b = 0;
  const MOD_ADLER = 65521;
  const end = offset + length;
  for (let i = offset; i < end; i++) {
    a = (a + buf[i]) % MOD_ADLER;
    b = (b + a) % MOD_ADLER;
  }
  return ((b << 16) | a) >>> 0;
}

/**
 * Encode an activation tensor into a binary MESH packet
 * @param {Object} params
 * @param {BigInt|number} params.sessionId
 * @param {number} params.seq
 * @param {Float32Array} params.activations - flat float32 array
 * @param {Array<number>} params.shape - e.g. [1, seqLen, 384]
 * @returns {ArrayBuffer}
 */
function encodeActivation({ sessionId, seq, activations, shape, flags = 0 }) {
  if (!(activations instanceof Float32Array)) {
    throw new TypeError("activations must be a Float32Array");
  }
  const payloadBytes = activations.byteLength;
  const totalSize = HEADER_SIZE + payloadBytes;
  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);
  const u8 = new Uint8Array(buffer);

  // 0..3: Magic
  view.setUint32(0, MAGIC, false);
  // 4: Version
  view.setUint8(4, PROTOCOL_VERSION);
  // 5: MsgType
  view.setUint8(5, MSG_TYPES.ACTIVATION);
  // 6..7: Flags
  view.setUint16(6, flags, false);
  // 8..15: Session ID
  view.setBigUint64(8, BigInt(sessionId || 0), false);
  // 16..19: Seq
  view.setUint32(16, seq, false);
  // 20: Data Type (FP32)
  view.setUint8(20, DATA_TYPES.FP32);
  // 21: Rank
  view.setUint8(21, shape.length);
  // 22..23: Reserved
  view.setUint16(22, 0, false);
  // 24..39: Shape (up to 4 dims)
  for (let i = 0; i < 4; i++) {
    view.setUint32(24 + i * 4, shape[i] || 0, false);
  }
  // 40..43: Payload length
  view.setUint32(40, payloadBytes, false);

  // Copy payload
  const payloadView = new Uint8Array(activations.buffer, activations.byteOffset, activations.byteLength);
  u8.set(payloadView, HEADER_SIZE);

  // 44..47: Checksum
  const checksum = adler32(u8, HEADER_SIZE, payloadBytes);
  view.setUint32(44, checksum, false);

  return buffer;
}

/**
 * Encode a generated token packet (Phone B -> Phone A)
 * @param {Object} params
 * @param {BigInt|number} params.sessionId
 * @param {number} params.seq
 * @param {number} params.token - byte value 0..255
 * @param {boolean} params.isEos
 * @param {Object} [params.timing] - execution stats
 * @param {Float32Array} [params.logits] - optional full logits (Stage 5 validation)
 * @returns {ArrayBuffer}
 */
function encodeToken({ sessionId, seq, token, isEos = false, timing = {}, logits = null }) {
  const flags = isEos ? 1 : 0;
  // Stage 5: logits (Float32Array) are optional, sent only when Device A asked (FLAG_WANT_LOGITS). float32 -> double -> JSON -> double -> float32 is exact.
  const body = { token, timing };
  if (logits) body.logits = Array.from(logits);
  const jsonStr = JSON.stringify(body);
  const encoder = new TextEncoder();
  const payloadU8 = encoder.encode(jsonStr);
  const payloadBytes = payloadU8.byteLength;

  const totalSize = HEADER_SIZE + payloadBytes;
  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);
  const u8 = new Uint8Array(buffer);

  view.setUint32(0, MAGIC, false);
  view.setUint8(4, PROTOCOL_VERSION);
  view.setUint8(5, MSG_TYPES.TOKEN);
  view.setUint16(6, flags, false);
  view.setBigUint64(8, BigInt(sessionId || 0), false);
  view.setUint32(16, seq, false);
  view.setUint8(20, DATA_TYPES.UTF8_JSON);
  view.setUint8(21, 1);
  view.setUint16(22, 0, false);
  view.setUint32(24, 1, false);
  view.setUint32(28, 0, false);
  view.setUint32(32, 0, false);
  view.setUint32(36, 0, false);
  view.setUint32(40, payloadBytes, false);

  u8.set(payloadU8, HEADER_SIZE);

  const checksum = adler32(u8, HEADER_SIZE, payloadBytes);
  view.setUint32(44, checksum, false);

  return buffer;
}

/**
 * Encode a control message (e.g. handshake, cancel, done)
 */
function encodeControl({ sessionId, seq = 0, command, data = {} }) {
  const jsonStr = JSON.stringify({ command, ...data });
  const payloadU8 = new TextEncoder().encode(jsonStr);
  const payloadBytes = payloadU8.byteLength;

  const totalSize = HEADER_SIZE + payloadBytes;
  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);
  const u8 = new Uint8Array(buffer);

  view.setUint32(0, MAGIC, false);
  view.setUint8(4, PROTOCOL_VERSION);
  view.setUint8(5, MSG_TYPES.CONTROL);
  view.setUint16(6, 0, false);
  view.setBigUint64(8, BigInt(sessionId || 0), false);
  view.setUint32(16, seq, false);
  view.setUint8(20, DATA_TYPES.UTF8_JSON);
  view.setUint8(21, 1);
  view.setUint16(22, 0, false);
  view.setUint32(24, payloadBytes, false);
  view.setUint32(28, 0, false);
  view.setUint32(32, 0, false);
  view.setUint32(36, 0, false);
  view.setUint32(40, payloadBytes, false);

  u8.set(payloadU8, HEADER_SIZE);
  const checksum = adler32(u8, HEADER_SIZE, payloadBytes);
  view.setUint32(44, checksum, false);

  return buffer;
}

/**
 * Decode and validate any incoming binary MESH packet
 * @param {ArrayBuffer|Uint8Array} rawData
 * @returns {Object} Decoded packet
 */
function decodePacket(rawData) {
  let buffer;
  let byteOffset = 0;
  if (rawData instanceof ArrayBuffer) {
    buffer = rawData;
  } else if (ArrayBuffer.isView(rawData)) {
    buffer = rawData.buffer;
    byteOffset = rawData.byteOffset;
  } else {
    throw new Error("Invalid raw packet type: expected ArrayBuffer or TypedArray");
  }

  if (buffer.byteLength - byteOffset < HEADER_SIZE) {
    throw new Error(`Packet too small: got ${buffer.byteLength - byteOffset} bytes, minimum ${HEADER_SIZE}`);
  }

  const view = new DataView(buffer, byteOffset);
  const magic = view.getUint32(0, false);
  if (magic !== MAGIC) {
    throw new Error(`Invalid protocol magic: 0x${magic.toString(16)} != 0x${MAGIC.toString(16)}`);
  }

  const version = view.getUint8(4);
  if (version !== PROTOCOL_VERSION) {
    throw new Error(`Unsupported protocol version: ${version}`);
  }

  const msgType = view.getUint8(5);
  const flags = view.getUint16(6, false);
  const sessionId = view.getBigUint64(8, false);
  const seq = view.getUint32(16, false);
  const dataType = view.getUint8(20);
  const rank = view.getUint8(21);
  if (rank > MAX_RANK) {
    throw new Error(`Invalid tensor rank ${rank}: header holds at most ${MAX_RANK} dimensions`);
  }

  const shape = [];
  for (let i = 0; i < rank; i++) {
    shape.push(view.getUint32(24 + i * 4, false));
  }

  const payloadLength = view.getUint32(40, false);
  const expectedChecksum = view.getUint32(44, false);

  if (buffer.byteLength - byteOffset < HEADER_SIZE + payloadLength) {
    throw new Error(`Truncated packet: expected ${HEADER_SIZE + payloadLength} bytes, have ${buffer.byteLength - byteOffset}`);
  }

  const u8 = new Uint8Array(buffer, byteOffset);
  const actualChecksum = adler32(u8, HEADER_SIZE, payloadLength);
  if (actualChecksum !== expectedChecksum) {
    throw new Error(`Packet checksum mismatch: expected ${expectedChecksum}, computed ${actualChecksum}`);
  }

  // Parse payload according to type
  let payload = null;
  if (msgType === MSG_TYPES.ACTIVATION && dataType === DATA_TYPES.FP32) {
    // Zero-copy Float32Array view directly over buffer
    const numFloats = payloadLength / 4;
    payload = new Float32Array(buffer, byteOffset + HEADER_SIZE, numFloats);
  } else if (dataType === DATA_TYPES.UTF8_JSON && msgType !== MSG_TYPES.TENSOR) {
    const jsonBytes = new Uint8Array(buffer, byteOffset + HEADER_SIZE, payloadLength);
    const jsonStr = new TextDecoder().decode(jsonBytes);
    payload = JSON.parse(jsonStr);
  } else {
    payload = new Uint8Array(buffer, byteOffset + HEADER_SIZE, payloadLength);
  }

  return {
    magic,
    version,
    msgType,
    flags,
    isEos: (flags & 0x01) !== 0,
    sessionId: sessionId.toString(),
    seq,
    dataType,
    rank,
    shape,
    payloadLength,
    payload,
  };
}

/**
 * Validate a shape and return its element count (product of dimensions).
 * Rank 0 is a scalar (1 element). Any zero dimension gives an empty tensor (0 elements).
 */
function shapeNumel(shape) {
  if (!Array.isArray(shape)) throw new TypeError("shape must be an array of integers");
  if (shape.length > MAX_RANK) throw new RangeError(`Tensor rank ${shape.length} exceeds the maximum of ${MAX_RANK}`);
  let numel = 1;
  for (const d of shape) {
    if (!Number.isInteger(d) || d < 0 || d > 0xffffffff) {
      throw new RangeError(`Invalid dimension ${d}: must be an integer between 0 and 4294967295`);
    }
    numel *= d;
    if (!Number.isSafeInteger(numel)) throw new RangeError("Tensor shape is too large");
  }
  return numel;
}

function requireLittleEndian() {
  if (!IS_LITTLE_ENDIAN) throw new Error("Tensor payloads are little-endian; this host is big-endian and is not supported");
}

/**
 * Serialize a tensor into one self-describing MESH packet (message type TENSOR).
 * Reuses encodeActivation for the 48-byte header + payload layout; this adds the
 * validation encodeActivation lacks, and tags the packet as TENSOR instead of ACTIVATION.
 * @param {Object} params
 * @param {Float32Array} params.data - flat, row-major elements
 * @param {Array<number>} params.shape - 0..4 non-negative integers whose product === data.length
 * @returns {ArrayBuffer} one contiguous packet (one copy of the data is made into it)
 */
function encodeTensor({ sessionId = 0, seq = 0, data, shape, flags = 0 }) {
  requireLittleEndian();
  if (!(data instanceof Float32Array)) {
    const kind = data && data.constructor ? data.constructor.name : typeof data;
    throw new TypeError(`Unsupported tensor type ${kind}: only Float32Array is supported`);
  }
  const numel = shapeNumel(shape);
  if (data.length !== numel) {
    throw new RangeError(`Shape [${shape.join(", ")}] holds ${numel} elements but data has ${data.length}`);
  }
  if (!Number.isInteger(seq) || seq < 0 || seq > 0xffffffff) throw new RangeError(`Invalid seq ${seq}`);
  if (data.byteLength > 0xffffffff) throw new RangeError("Tensor payload exceeds 4 GiB");

  const buffer = encodeActivation({ sessionId, seq, activations: data, shape, flags });
  // The checksum covers only the payload, so changing the message-type byte keeps it valid.
  new DataView(buffer).setUint8(5, MSG_TYPES.TENSOR);
  return buffer;
}

/**
 * Parse and validate a TENSOR packet, and rebuild the typed array.
 * @param {ArrayBuffer|ArrayBufferView} rawData - exactly one packet
 * @param {Object} [opts]
 * @param {boolean} [opts.copy=false] - true forces data to own its memory. By default data
 *   is a zero-copy view over rawData's buffer when 4-byte aligned (copied only if it is not).
 * @returns {{data: Float32Array, shape: number[], dtype: string, numel: number, sessionId: string, seq: number, flags: number}}
 */
function decodeTensor(rawData, { copy = false } = {}) {
  requireLittleEndian();
  const p = decodePacket(rawData); // magic, version, rank, truncation, payload checksum
  if (p.msgType !== MSG_TYPES.TENSOR) {
    throw new Error(`Not a tensor packet: message type 0x${p.msgType.toString(16).padStart(2, "0")}`);
  }
  const dt = TENSOR_DTYPES[p.dataType];
  if (!dt) throw new Error(`Unsupported tensor dtype code 0x${p.dataType.toString(16).padStart(2, "0")}`);

  const numel = shapeNumel(p.shape);
  const expected = numel * dt.bytes;
  if (p.payloadLength !== expected) {
    throw new Error(`Payload length mismatch: shape [${p.shape.join(", ")}] of ${dt.name} needs ${expected} bytes, header says ${p.payloadLength}`);
  }
  if (rawData.byteLength !== HEADER_SIZE + p.payloadLength) {
    throw new Error(`Packet size mismatch: expected ${HEADER_SIZE + p.payloadLength} bytes, got ${rawData.byteLength}`);
  }

  const bytes = p.payload; // Uint8Array view over the payload bytes (no copy yet)
  let data;
  if (!copy && bytes.byteOffset % dt.bytes === 0) {
    data = new dt.ctor(bytes.buffer, bytes.byteOffset, numel); // zero-copy view
  } else {
    const owned = new Uint8Array(bytes.byteLength); // fresh, aligned buffer
    owned.set(bytes);
    data = new dt.ctor(owned.buffer, 0, numel);
  }
  return { data, shape: p.shape, dtype: dt.name, numel, sessionId: p.sessionId, seq: p.seq, flags: p.flags };
}

/* =============================================================================
 * Stage 5: activation transfer helpers (additive; nothing above this line changed
 * except encodeToken, which gained an optional `logits` field).
 * ========================================================================== */

// Flags (header bytes 6..7). Bit 0 (is_eos) already existed.
const FLAG_WANT_LOGITS = 0x0002; // ACTIVATION: ask Device B to include the full logits in its reply (validation runs)

/** Error with a stable machine-readable `code`; travels over the wire in ERROR packets. */
class MeshError extends Error {
  constructor(code, message) {
    super(message);
    this.name = "MeshError";
    this.code = code;
  }
}

/** ERROR packet (MSG_TYPES.ERROR). sessionId 0 means "could not attribute this to a request". */
function encodeError({ sessionId = 0, seq = 0, code, message = "" }) {
  const payloadU8 = new TextEncoder().encode(JSON.stringify({ code: String(code), message: String(message) }));
  const buffer = new ArrayBuffer(HEADER_SIZE + payloadU8.byteLength);
  const view = new DataView(buffer);
  view.setUint32(0, MAGIC, false);
  view.setUint8(4, PROTOCOL_VERSION);
  view.setUint8(5, MSG_TYPES.ERROR);
  view.setUint16(6, 0, false);
  view.setBigUint64(8, BigInt(sessionId || 0), false);
  view.setUint32(16, seq, false);
  view.setUint8(20, DATA_TYPES.UTF8_JSON);
  view.setUint8(21, 1);
  view.setUint32(24, payloadU8.byteLength, false);
  view.setUint32(40, payloadU8.byteLength, false);
  new Uint8Array(buffer).set(payloadU8, HEADER_SIZE);
  view.setUint32(44, adler32(new Uint8Array(buffer), HEADER_SIZE, payloadU8.byteLength), false);
  return buffer;
}

/**
 * Check a decoded ACTIVATION packet BEFORE any compute. decodePacket only proves the bytes
 * arrived intact (magic, length, checksum); it does not know what an activation must look like.
 * @returns {{T: number}} the sequence length
 */
function validateActivation(p, { dModel = 384, maxSeqLen = 512 } = {}) {
  if (p.msgType !== MSG_TYPES.ACTIVATION) throw new MeshError("BAD_PACKET", `expected an ACTIVATION packet, got message type 0x${p.msgType.toString(16)}`);
  if (p.dataType !== DATA_TYPES.FP32) throw new MeshError("BAD_DTYPE", `activation must be float32 (0x01), got dtype 0x${p.dataType.toString(16)}`);
  if (p.rank !== 3) throw new MeshError("BAD_SHAPE", `activation must have rank 3 [1, T, ${dModel}], got rank ${p.rank}`);
  const [b, T, d] = p.shape;
  if (b !== 1) throw new MeshError("BAD_SHAPE", `batch dimension must be 1, got ${b}`);
  if (d !== dModel) throw new MeshError("BAD_SHAPE", `hidden dimension must be ${dModel}, got ${d}`);
  if (!Number.isInteger(T) || T < 1 || T > maxSeqLen) throw new MeshError("BAD_SHAPE", `sequence length must be 1..${maxSeqLen}, got ${T}`);
  if (p.payloadLength !== T * d * 4) throw new MeshError("BAD_SHAPE", `shape [1, ${T}, ${d}] needs ${T * d * 4} payload bytes, header says ${p.payloadLength}`);
  const x = p.payload;
  if (!(x instanceof Float32Array) || x.length !== T * d) throw new MeshError("BAD_PACKET", "activation payload is not a Float32Array of the declared size");
  for (let i = 0; i < x.length; i++) {
    if (!Number.isFinite(x[i])) throw new MeshError("NON_FINITE", `activation contains NaN/Infinity at element ${i}`);
  }
  return { T };
}

/* ---- Chunking: DataChannel messages are capped (pc.sctp.maxMessageSize), a T=512 activation is 786,480 bytes ----
 * A finished MESH packet is cut into slices; each slice is wrapped in a 20-byte chunk header:
 *   0..3 "CHNK" | 4..7 messageId | 8..11 chunkIndex | 12..15 chunkCount | 16..19 totalBytes (whole packet) | payload
 * The receiver rebuilds the EXACT original bytes, then decodePacket (checksum, validation) runs on the whole.
 * Chunk frames start with "CHNK", which can never be mistaken for a MESH packet or for "TEST" traffic. */
const CHUNK_MAGIC = 0x43484e4b; // "CHNK"
const CHUNK_HEADER_SIZE = 20;
const MAX_REASSEMBLED_BYTES = 8 * 1024 * 1024; // far above one full-context activation; bounds memory from a bad peer
const MIN_MESSAGE_SIZE = 256;

function isChunk(raw) {
  return raw instanceof ArrayBuffer && raw.byteLength >= CHUNK_HEADER_SIZE && new DataView(raw).getUint32(0, false) === CHUNK_MAGIC;
}

/** @returns {ArrayBuffer[]} [packet] itself when it already fits, otherwise the chunk frames (each <= maxMessageSize) */
function splitIntoChunks(packet, maxMessageSize, messageId = 0) {
  if (!(packet instanceof ArrayBuffer)) throw new TypeError("packet must be an ArrayBuffer");
  if (!Number.isInteger(maxMessageSize) || maxMessageSize < MIN_MESSAGE_SIZE) throw new RangeError(`maxMessageSize ${maxMessageSize} is too small`);
  if (packet.byteLength <= maxMessageSize) return [packet];
  if (packet.byteLength > MAX_REASSEMBLED_BYTES) throw new MeshError("TOO_LARGE", `packet of ${packet.byteLength} bytes exceeds ${MAX_REASSEMBLED_BYTES}`);
  const per = maxMessageSize - CHUNK_HEADER_SIZE;
  const count = Math.ceil(packet.byteLength / per);
  const src = new Uint8Array(packet);
  const frames = [];
  for (let i = 0; i < count; i++) {
    const slice = src.subarray(i * per, Math.min((i + 1) * per, src.length));
    const frame = new ArrayBuffer(CHUNK_HEADER_SIZE + slice.length);
    const v = new DataView(frame);
    v.setUint32(0, CHUNK_MAGIC, false);
    v.setUint32(4, messageId >>> 0, false);
    v.setUint32(8, i, false);
    v.setUint32(12, count, false);
    v.setUint32(16, src.length, false);
    new Uint8Array(frame).set(slice, CHUNK_HEADER_SIZE);
    frames.push(frame);
  }
  return frames;
}

/** Rebuilds one packet from in-order chunk frames. Any inconsistency throws and discards the partial message. */
class ChunkReassembler {
  constructor({ maxBytes = MAX_REASSEMBLED_BYTES } = {}) { this.maxBytes = maxBytes; this.state = null; this.discarded = 0; this.poisonId = null; this.suppressed = 0; }
  get inProgress() { return this.state !== null; }
  // Stage 6: reset() drops a partial message AND forgets which broken message to ignore (used when the owner expires or disposes the transfer).
  reset() { this.state = null; this.poisonId = null; }
  /** @returns {ArrayBuffer|null} the complete packet on the last chunk, otherwise null */
  push(raw) {
    let id = null;
    // Stage 6: remember WHICH message was broken, so its remaining frames (still in flight) are ignored instead of each raising another error.
    const fail = (msg) => { this.state = null; if (id !== null) this.poisonId = id; throw new MeshError("BAD_CHUNK", msg); };
    if (!isChunk(raw)) fail("not a chunk frame");
    const v = new DataView(raw);
    id = v.getUint32(4, false);
    const index = v.getUint32(8, false), count = v.getUint32(12, false), total = v.getUint32(16, false);
    if (this.poisonId === id && index !== 0) { this.suppressed++; return null; }   // leftover frame of a message we already rejected
    if (index === 0 && this.poisonId !== null) this.poisonId = null;               // a new message begins: stop ignoring
    const len = raw.byteLength - CHUNK_HEADER_SIZE;
    if (count < 2 || count > 65536) fail(`invalid chunk count ${count}`);
    if (total < 1 || total > this.maxBytes) fail(`declared total of ${total} bytes is outside 1..${this.maxBytes}`);
    if (len < 1) fail("empty chunk");
    if (index === 0) {
      if (this.state) { this.discarded++; this.state = null; }   // sender abandoned the previous message (e.g. it timed out); the new one is still fully checked
      this.state = { id, count, total, next: 0, offset: 0, buf: new Uint8Array(total) };
    }
    const s = this.state;
    if (!s) fail(`chunk ${index} arrived with no message in progress`);
    if (id !== s.id || count !== s.count || total !== s.total) fail("chunk header does not match the message in progress");
    if (index !== s.next) fail(`chunk ${index} arrived but chunk ${s.next} was expected`);
    if (s.offset + len > s.total) fail("chunks exceed the declared total size");
    s.buf.set(new Uint8Array(raw, CHUNK_HEADER_SIZE, len), s.offset);
    s.offset += len;
    s.next++;
    if (s.next < s.count) return null;
    if (s.offset !== s.total) fail(`message ended with ${s.offset} of ${s.total} bytes`);
    this.state = null;
    return s.buf.buffer;
  }
}

// Export for Node and Browser environments
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    MAGIC,
    PROTOCOL_VERSION,
    MSG_TYPES,
    DATA_TYPES,
    HEADER_SIZE,
    adler32,
    encodeActivation,
    encodeToken,
    encodeControl,
    decodePacket,
    MAX_RANK,
    TENSOR_DTYPES,
    shapeNumel,
    encodeTensor,
    decodeTensor,
    FLAG_WANT_LOGITS,
    MeshError,
    encodeError,
    validateActivation,
    CHUNK_MAGIC,
    CHUNK_HEADER_SIZE,
    MAX_REASSEMBLED_BYTES,
    isChunk,
    splitIntoChunks,
    ChunkReassembler,
  };
} else if (typeof window !== "undefined") {
  window.MeshProtocol = {
    MAGIC,
    PROTOCOL_VERSION,
    MSG_TYPES,
    DATA_TYPES,
    HEADER_SIZE,
    adler32,
    encodeActivation,
    encodeToken,
    encodeControl,
    decodePacket,
    MAX_RANK,
    TENSOR_DTYPES,
    shapeNumel,
    encodeTensor,
    decodeTensor,
    FLAG_WANT_LOGITS,
    MeshError,
    encodeError,
    validateActivation,
    CHUNK_MAGIC,
    CHUNK_HEADER_SIZE,
    MAX_REASSEMBLED_BYTES,
    isChunk,
    splitIntoChunks,
    ChunkReassembler,
  };
}
