"use strict";
/**
 * MINIMOL Stage 6: the connection state machine.
 *
 * Pure logic, no browser APIs, so Node can test it (tests/test_stage6_reliability.js) and main.js can use it.
 *
 * Why it exists: before Stage 6 the question "are we connected?" was answered by five different variables
 * (S.live, S.stage, S.dc.readyState, S.pipe, S.pc.connectionState) that could disagree. Now there is ONE state,
 * and it can only change along the arrows in TRANSITIONS. An event that has no arrow from the current state is
 * refused (send() returns false). That is what makes late or duplicated callbacks harmless.
 *
 *   DISCONNECTED --START--> CONNECTING --DC_OPEN--> CONNECTED --PIPELINE_READY--> INFERENCE_READY
 *                               |                      |                              |
 *                               |                      +------------ LINK_LOST -------+--> RECONNECTING --DC_OPEN--> CONNECTED
 *                               +--FAIL--> FAILED <--FAIL-- (from any state that is connecting or connected)
 *   CANCEL / RESET: from any state back to DISCONNECTED.      FAILED leaves only through RESET.
 */
(function (root) {
  const STATES = ["DISCONNECTED", "CONNECTING", "CONNECTED", "INFERENCE_READY", "RECONNECTING", "FAILED"];

  const TRANSITIONS = {
    DISCONNECTED:    { START: "CONNECTING", RESET: "DISCONNECTED", CANCEL: "DISCONNECTED" },
    CONNECTING:      { DC_OPEN: "CONNECTED", FAIL: "FAILED", CANCEL: "DISCONNECTED", RESET: "DISCONNECTED" },
    CONNECTED:       { PIPELINE_READY: "INFERENCE_READY", LINK_LOST: "RECONNECTING", FAIL: "FAILED", CANCEL: "DISCONNECTED", RESET: "DISCONNECTED" },
    INFERENCE_READY: { PIPELINE_DROPPED: "CONNECTED", LINK_LOST: "RECONNECTING", FAIL: "FAILED", CANCEL: "DISCONNECTED", RESET: "DISCONNECTED" },
    RECONNECTING:    { DC_OPEN: "CONNECTED", FAIL: "FAILED", CANCEL: "DISCONNECTED", RESET: "DISCONNECTED" },
    FAILED:          { RESET: "DISCONNECTED" },
  };

  /** The state an event leads to, or null when that event is not allowed in this state. */
  function nextState(state, event) {
    const row = TRANSITIONS[state];
    return row && Object.prototype.hasOwnProperty.call(row, event) ? row[event] : null;
  }

  function createConnectionMachine({ onChange = null } = {}) {
    let state = "DISCONNECTED";
    const history = [];                                   // [{from, event, to}] kept short; useful when a test or a bug report needs "how did we get here?"
    return {
      get state() { return state; },
      history,
      is(...names) { return names.includes(state); },
      /** Apply an event. Returns true if the state changed or the event was a legal no-op arrow; false if it was refused. */
      send(event, info) {
        const to = nextState(state, event);
        if (to === null) return false;                    // refused: stale/duplicate/out-of-order event
        const from = state; state = to;
        history.push({ from, event, to }); if (history.length > 50) history.shift();
        if (onChange && (from !== to || event === "RESET")) { try { onChange(to, from, event, info); } catch (e) { /* a UI callback must not break the machine */ } }
        return true;
      },
    };
  }

  /**
   * Bounded reconnection with exponential backoff: attempt n waits baseMs * 2^(n-1).
   * Returns { retry: false } once `attempts` (already made) reaches `max`. There is deliberately no "retry forever".
   */
  function reconnectPlan({ attempts, max, baseMs }) {
    if (!(attempts < max)) return { retry: false, attempt: attempts, delayMs: 0 };
    return { retry: true, attempt: attempts + 1, delayMs: baseMs * Math.pow(2, attempts) };
  }

  const api = { STATES, TRANSITIONS, nextState, createConnectionMachine, reconnectPlan };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.MeshConnState = api;
})(typeof window !== "undefined" ? window : globalThis);
