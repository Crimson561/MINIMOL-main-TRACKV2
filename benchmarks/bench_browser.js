#!/usr/bin/env node
/**
 * Stage 7 — drive benchmarks/browser/bench.html in headless Chromium (one page, two RTCPeerConnections, real RTCDataChannel).
 *   node benchmarks/bench_browser.js <label> [engine=current|baseline] [iters]
 * Evidence level: "browser, same machine, loopback ICE". NOT two tabs, NOT two physical devices, NOT a real network.
 */
"use strict";
const http = require("http"), fs = require("fs"), path = require("path");
const ROOT = path.join(__dirname, ".."), { summarize, bootstrapMedianDiffCI } = require("./lib/stats"), { describeEnvironment } = require("./lib/env");
const CHROME = process.env.CHROME_PATH;
if (!CHROME) { console.error("Set CHROME_PATH to a Chrome/Chromium executable (and install puppeteer-core). See BENCHMARKS.md section 11."); process.exit(1); }
let puppeteer; try { puppeteer = require(process.env.PUPPETEER_CORE_PATH || "puppeteer-core"); } catch (e) { console.error("puppeteer-core is not installed (npm i puppeteer-core), or set PUPPETEER_CORE_PATH."); process.exit(1); }
const label = process.argv[2] || "browser", engine = process.argv[3] || "current", iters = process.argv[4] || "8";
const MAP = { "/protocol/": path.join(ROOT, "protocol"), "/engine/": path.join(ROOT, "engine"), "/baseline/": path.join(__dirname, "baseline"), "/weights/": path.join(ROOT, "room-server", "public", "weights") };
const server = http.createServer((req, res) => {
  const u = new URL(req.url, "http://x"); let file = null;
  if (u.pathname === "/" || u.pathname === "/bench.html") file = path.join(__dirname, "browser", "bench.html");
  for (const [pre, dir] of Object.entries(MAP)) if (u.pathname.startsWith(pre)) file = path.join(dir, path.basename(u.pathname));
  if (!file || !fs.existsSync(file)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "content-type": file.endsWith(".js") ? "text/javascript" : file.endsWith(".html") ? "text/html" : file.endsWith(".json") ? "application/json" : "application/octet-stream" }); fs.createReadStream(file).pipe(res);
});
(async () => {
  await new Promise((ok) => server.listen(0, "127.0.0.1", ok)); const port = server.address().port;
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: "shell", args: ["--no-sandbox", "--disable-features=WebRtcHideLocalIpsWithMdns", "--js-flags=--expose-gc"] });
  const page = await browser.newPage(); page.on("pageerror", (e) => console.error("[page error]", e.message)); page.on("console", (m) => { if (m.type() === "error") console.error("[console]", m.text()); });
  await page.goto(`http://127.0.0.1:${port}/bench.html?engine=${engine}&iters=${iters}`);
  await page.waitForFunction("window.__done === true", { timeout: 20 * 60 * 1000, polling: 2000 });
  const r = await page.evaluate("window.__result"); const version = await browser.version(); await browser.close(); server.close();
  const out = { schema: "minimol-stage7-browser/1", label, engineUnderTest: engine, browser: version, evidenceLevel: "browser-single-page-two-RTCPeerConnections-real-RTCDataChannel (same machine, loopback ICE; not two tabs, not physical devices, no real network)", environment: describeEnvironment(ROOT), raw: r };
  if (!r.ok) { console.error("FAILED:", r.fatal); process.exitCode = 1; }
  else {
    out.rawChannel = { negotiatedMaxMessageSize: r.rawChannel.negotiatedMaxMessageSize, rttMs: summarize(r.rawChannel.pingPong.rttMs),
      transfers: r.rawChannel.transfers.map((t) => { const s = summarize(t.timesMs); return { bytes: t.bytes, frames: t.frames, ms: s, throughputMBps_median: (t.bytes / 1e6) / (s.median / 1000), throughputMBps_best: (t.bytes / 1e6) / (s.min / 1000) }; }) };
    out.comparisons = r.workloads.map((w) => { const L = w.local, D = w.distributed; const ci = bootstrapMedianDiffCI(L, D); const sl = summarize(L), sd = summarize(D);
      const row = { id: w.id, failures: w.errors, local: sl, distributed: sd, absDiffMs: sd.median - sl.median, pctDiff: ((sd.median - sl.median) / sl.median) * 100, ci95: ci, verdict: Math.min(L.length, D.length) < 8 ? "insufficient samples" : ci.excludesZero ? (sd.median > sl.median ? "distributed slower (outside noise)" : "distributed faster (outside noise)") : "no measurable difference" };
      if (w.rows.length && w.rows[0].wireAB !== undefined) { const m = (k) => summarize(w.rows.map((x) => x[k])).median; row.phasesMedianMs = w.id === "gen" ? { wireAB_sum: m("wireAB"), wireBA_sum: m("wireBA"), compute_sum: m("compute"), residual_sum: m("residual") } : { aPre: m("aPre"), aCompute: m("aCompute"), serialize: m("serialize"), chunk: m("chunk"), gapGlue: m("gapGlue"), gapBeforeSend: m("gapBeforeSend"), wireAB: m("wireAB"), bProcNonCompute: m("bProcNonCompute"), bCompute: m("bCompute"), wireBA: m("wireBA"), aFinish: m("aFinish"), residual: m("residual"), totalMs: m("totalMs"), frames: m("frames"), packetBytes: m("activationPacketBytes") }; }
      return row; });
  }
  const f = path.join(__dirname, "results", `${label}.browser.json`); fs.writeFileSync(f, JSON.stringify(out, null, 2)); console.log("wrote", path.relative(ROOT, f));
  if (out.rawChannel) { console.log("channel maxMessageSize", out.rawChannel.negotiatedMaxMessageSize, "| ping-pong RTT median", out.rawChannel.rttMs.median.toFixed(3), "ms"); for (const t of out.rawChannel.transfers) console.log(`  ${t.bytes} B in ${t.frames} frame(s): median ${t.ms.median.toFixed(2)} ms, ${t.throughputMBps_median.toFixed(1)} MB/s`); for (const c of out.comparisons) console.log(`  ${c.id}: local ${c.local.median.toFixed(1)} dist ${c.distributed.median.toFixed(1)} (${c.pctDiff.toFixed(1)}%) ${c.verdict} failures ${c.failures.length}`); }
})().catch((e) => { console.error(e); process.exit(2); });
