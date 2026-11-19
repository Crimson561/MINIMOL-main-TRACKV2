// End-to-end UI test: two browser pages, driven the way a PERSON uses the app (visible elements only, real clicks).
// Unlike stage6_two_page.js (which reads app internals), this fails if a control the user needs is not actually on screen.
//   needs: weights generated, `npm install playwright` + Chromium (optional dependency; see TESTING.md section 5)
//   node tests/browser/ui_user_flow.js [baseUrl]     (without baseUrl it starts room-server itself on a spare port)
const { spawn } = require("child_process");
const path = require("path");
const { chromium } = require("playwright");

let passed = 0, failed = 0;
const check = (name, ok, detail = "") => { ok ? passed++ : failed++; console.log(`  ${ok ? "✓" : "✗ FAIL"} ${name}${detail ? "   [" + detail + "]" : ""}`); };

(async () => {
  let server = null, base = process.argv[2];
  if (!base) {
    const port = 3100 + Math.floor(Math.random() * 800);
    server = spawn(process.execPath, [path.join(__dirname, "..", "..", "room-server", "server.js")], { env: { ...process.env, PORT: String(port) }, stdio: "ignore" });
    base = `http://localhost:${port}`;
    await new Promise((r) => setTimeout(r, 1500));
  }
  const browser = await chromium.launch({ args: ["--disable-features=WebRtcHideLocalIpsWithMdns"] });
  try {
    const page = async () => { const p = await (await browser.newContext({ viewport: { width: 1100, height: 800 } })).newPage(); p.errs = []; p.on("pageerror", (e) => p.errs.push(e.message)); return p; };
    const A = await page(), B = await page();
    await A.goto(base); await B.goto(base);
    for (const p of [A, B]) await p.locator('[data-go="matchmaker"]:visible').first().click();

    console.log("U1. Host creates a room");
    await A.click("#createBtn");
    await A.waitForFunction(() => /^MESH-/.test(document.getElementById("codeOut").textContent), null, { timeout: 15000 });
    const code = (await A.textContent("#codeOut")).trim();
    check("room code is VISIBLE on the host's screen (not just in the DOM)", await A.locator("#codeOut").isVisible(), code);
    check("Copy button is visible", await A.locator("#copyBtn").isVisible());
    check("host is not offered a second Create/Join while waiting", !(await A.locator("#createBtn").isVisible()) && !(await A.locator("#joinBtn").isVisible()));
    check("status says it is waiting for the other device", /Waiting/.test(await A.textContent("#mmStatus")));

    console.log("U2. Joiner enters the code");
    await B.fill("#codeIn", code); await B.click("#joinBtn");
    for (const [n, p] of [["A", A], ["B", B]]) {
      let ok = true; try { await p.waitForFunction(() => conn.state === "INFERENCE_READY", null, { timeout: 90000 }); } catch (e) { ok = false; }
      check(`device ${n} reaches INFERENCE_READY (weights verified, WebRTC open, handshake done)`, ok);
    }
    check("both devices show the match screen with a visible 'Enter workspace' button", (await A.locator("#enterBtn").isVisible()) && (await B.locator("#enterBtn").isVisible()));

    console.log("U3. Run inference from the host");
    await A.click("#enterBtn"); await B.click("#enterBtn");
    check("workspace banner shows a live session on both devices", /Live session/.test(await A.textContent("#banner")) && /Live session/.test(await B.textContent("#banner")));
    await A.locator("button[data-view='inference']").click();
    check("untrained-model notice is visible next to the prompt", await A.locator("#untrainedNote").isVisible());
    await A.fill("#prompt", "MESH"); await A.click("#runBtn");
    let done = true; try { await A.waitForFunction(() => document.getElementById("genStatus").textContent.trim() === "Done", null, { timeout: 120000 }); } catch (e) { done = false; }
    check("generation finishes ('Done')", done, done ? "" : await A.textContent("#genStatus"));
    check("some output was produced", (await A.textContent("#out")).length > 0);
    check("tokens counter is non-zero", Number(await A.textContent("#mTok")) > 0, await A.textContent("#mTok"));
    check("no uncaught page errors on either device", A.errs.length + B.errs.length === 0, [...A.errs, ...B.errs].join(" | "));
  } catch (e) { failed++; console.log("  ✗ test aborted:", e.message.split("\n")[0]); }
  await browser.close(); if (server) server.kill();
  console.log(failed ? `\nUI USER-FLOW TEST FAILED: ${passed} passed, ${failed} failed` : `\nUI USER-FLOW TEST PASSED: ${passed} passed, 0 failed`);
  process.exit(failed ? 1 : 0);
})();
