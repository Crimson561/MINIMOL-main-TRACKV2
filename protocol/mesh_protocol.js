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
 */

"use strict";

const MAGIC = 0x4d455348; // "MESH"
const PROTOCOL_VERSION = 0x01;

const MSG_TYPES = {
  ACTIVATION: 0x01,
  TOKEN: 0x02,
  CONTROL: 0x03,
  ERROR: 0x04,
};

const DATA_TYPES = {
  FP32: 0x01,
  UINT8: 0x02,
  INT32: 0x03,
  UTF8_JSON: 0x04,
};

const HEADER_SIZE = 48;

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
  } else if (dataType === DATA_TYPES.UTF8_JSON) {
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
  };
}
