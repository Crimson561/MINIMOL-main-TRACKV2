/**
 * OPTIONAL scripted browser check for Stage 6 (NOT part of the Node suites, NOT a project dependency).
 *
 * Drives TWO headless Chromium pages on ONE machine through the real room-server (Socket.IO) and a real WebRTC DataChannel.
 * Evidence level: "scripted two-page". It is NOT a two-tab run done by hand and NOT a physical-device test.
 *
 * Needs: weights exported, `cd room-server && npm install`, and Playwright available to Node (npm i -g playwright, or similar).
 * Run:   node tests/browser/stage6_two_page.js        (it starts and stops room-server itself; port 3000 must be free)
 */
"use strict";
const { spawn } = require("child_process");
const path = require("path");
const { chromium } = require("playwright");

const ROOT = path.join(__dirname, "..", "..");
const URL = "http://localhost:3000/";
let pass = 0, fail = 0;
const ok = (c, name, extra = "") => { if (c) { pass++; console.log("  ✓ " + name + (extra ? "   [" + extra + "]" : "")); } else { fail++; console.log("  ✗ " + name + (extra ? "   [" + extra + "]" : "")); } };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const snap = (p) => p.evaluate(() => ({ state: conn.state, gen: S.gen, timers: S.timers.size, reconnects: S.reconnects, pipe: !!S.pipe, dc: S.dc && S.dc.readyState, status: document.getElementById("genStatus").textContent, failMsg: document.getElementById("failMsg").textContent, screen: document.body.dataset.screen }));
async function waitState(p, want, ms, label) { const t0 = Date.now(); for (;;) { const s = await snap(p); if (s.state === want) return Date.now() - t0; if (Date.now() - t0 > ms) throw new Error(`${label}: wanted ${want} within ${ms} ms, last ${JSON.stringify(s)}`); await sleep(100); } }

async function pair(browser) {
  const A = await (await browser.newContext()).newPage(), B = await (await browser.newContext()).newPage();
  const errors = []; for (const [n, p] of [["A", A], ["B", B]]) p.on("pageerror", (e) => errors.push(n + ": " + e.message));
  await A.goto(URL); await B.goto(URL);
  for (const p of [A, B]) await p.locator('[data-go="matchmaker"]:visible').first().click();   // landing -> matchmaker screen
  await A.click("#createBtn"); await A.waitForFunction(() => document.getElementById("codeOut").textContent.startsWith("MESH-"));
  const code = await A.textContent("#codeOut"); await B.fill("#codeIn", code); await B.click("#joinBtn");
  await waitState(A, "INFERENCE_READY", 90000, "A first connect"); await waitState(B, "INFERENCE_READY", 90000, "B first connect");
  return { A, B, errors };
}
async function runOnA(A, prompt = "MESH") { await A.evaluate((t) => { document.getElementById("prompt").value = t; document.getElementById("runBtn").click(); }, prompt); }
async function genDone(A, ms = 90000) { await A.waitForFunction(() => !S.running, null, { timeout: ms }); return (await snap(A)).status; }

(async () => {
  const server = spawn("node", ["server.js"], { cwd: path.join(ROOT, "room-server"), stdio: "ignore" }); await sleep(1500);
  const browser = await chromium.launch({ args: ["--disable-features=WebRtcHideLocalIpsWithMdns"] });
  try {
    console.log("\nS0. Normal flow still works (Stage 5 regression)");
    let { A, B, errors } = await pair(browser);
    await runOnA(A, "MESH"); let st = await genDone(A); ok(st === "Done", "inference completes through the real DataChannel", st);
    ok((await A.textContent("#out")).length > 0, "output text produced");

    console.log("\nS1. DataChannel closed on Device A during inference");
    await runOnA(A, "MESH MESH MESH"); await sleep(250);
    const t0 = Date.now(); await A.evaluate(() => S.dc.close());
    st = await genDone(A, 15000); const failMs = Date.now() - t0;
    ok(/closed|replaced|lost|reconnect/i.test(st) && st !== "Done", "request fails with a connection error (not a hang, not 'Done')", `"${st}" after ${failMs} ms`);
    ok(failMs < 5000, "failure is reported in seconds, far below the 20 s timeout");
    const mid = await snap(A); ok(["RECONNECTING", "CONNECTED", "INFERENCE_READY"].includes(mid.state), "A did not stay in a fake 'connected' state", mid.state);
    const tRe = await waitState(A, "INFERENCE_READY", 60000, "A reconnect"); await waitState(B, "INFERENCE_READY", 60000, "B reconnect");
    ok(true, "both devices reconnected over a NEW peer connection and are INFERENCE_READY again", tRe + " ms");
    const g = await snap(A); ok(g.reconnects >= 1, "reconnect attempt counted (bounded retries)", "reconnects=" + g.reconnects);
    await runOnA(A, "MESH"); st = await genDone(A); ok(st === "Done", "inference works again on the new session", st);
    ok((await snap(A)).reconnects === 0, "a successful round trip refills the retry budget");
    const idle = await snap(A); ok(idle.timers <= 1, "no stray timers left running on A", "timers=" + idle.timers);
    ok(errors.length === 0, "no uncaught page errors", errors.join(" | "));
    await A.context().close(); await B.context().close();

    console.log("\nS2. Cancel / cleanup / duplicate start");
    const C = await (await browser.newContext()).newPage(); await C.goto(URL); await C.locator('[data-go="matchmaker"]:visible').first().click();
    await C.click("#createBtn"); await C.waitForFunction(() => document.getElementById("codeOut").textContent.startsWith("MESH-"));
    const dup = await C.evaluate(async () => { try { await connect("A"); return "no error"; } catch (e) { return e.message; } });
    ok(/already in progress/.test(dup), "a second connect() while one is running is refused", dup);
    await C.click("#cancelBtn"); let c = await C.evaluate(() => ({ s: conn.state, timers: S.timers.size, sock: S.socket, pc: S.pc, dc: S.dc }));
    ok(c.s === "DISCONNECTED" && c.timers === 0 && !c.sock && !c.pc && !c.dc, "Cancel leaves no socket, peer connection or timer behind", JSON.stringify({ s: c.s, timers: c.timers }));
    await C.evaluate(() => { reset(); reset(); teardown(); teardown(); });
    c = await C.evaluate(() => conn.state); ok(c === "DISCONNECTED", "cleanup called repeatedly is harmless");
    await C.context().close();

    console.log("\nS3. Device B's page disappears while connected");
    ({ A, B } = await pair(browser)); const t1 = Date.now(); await B.close();
    const left = await waitState(A, "FAILED", 60000, "A after B closed"); const f = await snap(A);
    ok(true, "A noticed the loss and ended in FAILED, not a permanent 'connected' UI", left + " ms");
    ok(/left the room/i.test(f.failMsg), "failure message explains why", f.failMsg); ok(f.screen === "matchmaker", "UI moved to the existing failure screen with Retry");
    ok(f.timers === 0 && !f.pipe && !f.dc, "nothing left running after the failure", JSON.stringify({ timers: f.timers, pipe: f.pipe }));
    await A.click("#retryBtn"); ok((await snap(A)).state === "DISCONNECTED", "Retry returns to a clean DISCONNECTED state");
    await A.context().close();

    console.log("\nS4. Reconnection is bounded (B never answers the new offers)");
    ({ A, B } = await pair(browser));
    await B.evaluate(() => S.socket.off(CONFIG.events.offer)); await A.evaluate(() => { CONFIG.reconnectEstablishMs = 3000; });
    const t2 = Date.now(); await A.evaluate(() => S.dc.close());
    const gaveUp = await waitState(A, "FAILED", 60000, "A giving up"); const f4 = await snap(A);
    ok(/Gave up after 2 reconnection attempt/.test(f4.failMsg), "A made exactly 2 attempts then stopped", f4.failMsg);
    ok(gaveUp < 30000, "total time is bounded", gaveUp + " ms"); await A.context().close(); await B.context().close().catch(() => {});
  } catch (e) { fail++; console.log("  ✗ SCRIPT ERROR: " + e.message); }
  await browser.close(); server.kill();
  console.log(`\nBROWSER (scripted, two headless pages, one machine): ${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
})();
