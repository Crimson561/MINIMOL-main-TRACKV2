/**
 * Stage 7 — an in-process A<->B link with NO artificial delay.
 *
 * tests/test_stage5_pipeline.js's Link uses setTimeout(..., 1): 1 ms of FAKE latency, fine for correctness tests, wrong for benchmarks.
 * This link delivers on the next event-loop turn (setImmediate) and copies each frame (a real channel also copies bytes into its own buffer).
 * It measures the software cost of the pipeline, NOT any network. Label every number from it "loopback (simulated, no network)".
 */
"use strict";

class LoopbackLink {
  constructor(maxMessageSize = 262144) {
    this.max = maxMessageSize; this.toA = null; this.toB = null; this.biggest = 0;
  }
  transport(dir) {
    const link = this;
    return {
      maxMessageSize: () => link.max,
      send(buf) {
        if (buf.byteLength > link.max) throw new RangeError(`message of ${buf.byteLength} bytes exceeds the channel limit ${link.max}`);
        link.biggest = Math.max(link.biggest, buf.byteLength);
        const copy = buf.slice(0);
        const target = dir === "AtoB" ? link.toB : link.toA;
        setImmediate(() => target(copy));
      },
    };
  }
}
module.exports = { LoopbackLink };
