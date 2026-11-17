// Stage 7 — profile workload for LONG sequences (T=512): 1 warm-up + 3 local forward passes. Use with --cpu-prof, see profile_workload.js.
const { loadModel, fixedTokens } = require("./lib/model"); const m = loadModel(); const t = fixedTokens(512);
m.full.forward(t); for (let i = 0; i < 3; i++) m.full.forward(t); console.log("profiled T=512 x3 (+1 warm-up)");
