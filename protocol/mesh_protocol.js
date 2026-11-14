/**
 * MESH Binary Protocol for WebRTC DataChannel
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
 * @returns {ArrayBuffer}
 */
function encodeToken({ sessionId, seq, token, isEos = false, timing = {} }) {
  const flags = isEos ? 1 : 0;
  const jsonStr = JSON.stringify({ token, timing });
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
  };
}
