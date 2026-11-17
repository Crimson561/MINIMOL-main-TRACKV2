#!/usr/bin/env node
/**
 * Stage 7 — read a .cpuprofile (Chrome DevTools / V8 format) and print where CPU time went.
 *   "self"  = time spent inside the function's own code (not in functions it called). This is what finds a hot loop.
 *   "total" = self + everything it called. This is what finds an expensive phase.
 */
"use strict";
const fs = require("fs"), path = require("path");
const file = process.argv[2]; if (!file) { console.error("usage: analyze_profile.js <file.cpuprofile> [topN]"); process.exit(1); }
const top = Number(process.argv[3] || 14);
const prof = JSON.parse(fs.readFileSync(file, "utf-8"));
const byId = new Map(prof.nodes.map((n) => [n.id, n]));
const parent = new Map(); for (const n of prof.nodes) for (const c of n.children || []) parent.set(c, n.id);
const self = new Map();
for (let i = 0; i < prof.samples.length; i++) self.set(prof.samples[i], (self.get(prof.samples[i]) || 0) + (prof.timeDeltas[i] || 0));
const key = (n) => { const f = n.callFrame; const u = f.url ? path.basename(f.url) : ""; return `${f.functionName || "(anonymous)"} ${u}${u ? ":" + (f.lineNumber + 1) : ""}`; };
const selfBy = new Map(), totalBy = new Map(); let all = 0;
for (const [id, t] of self) {
  all += t; const n = byId.get(id); const k = key(n); selfBy.set(k, (selfBy.get(k) || 0) + t);
  const seen = new Set(); for (let cur = id; cur !== undefined; cur = parent.get(cur)) { const kk = key(byId.get(cur)); if (!seen.has(kk)) { seen.add(kk); totalBy.set(kk, (totalBy.get(kk) || 0) + t); } }
}
const show = (m, title) => { console.log(`\n${title}  (profile length ${(all / 1000).toFixed(0)} ms, ${prof.samples.length} samples)`); [...m].sort((a, b) => b[1] - a[1]).slice(0, top).forEach(([k, t]) => console.log(`${(t / all * 100).toFixed(1).padStart(6)}%  ${(t / 1000).toFixed(0).padStart(8)} ms  ${k}`)); };
show(selfBy, "SELF time by function"); show(totalBy, "TOTAL time by function (includes callees)");
