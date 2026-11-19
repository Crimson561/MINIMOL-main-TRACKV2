/** Stage 7 — record the environment a benchmark ran in. Only facts the runtime can report; nothing is guessed. */
"use strict";
const os = require("os");
const { execSync } = require("child_process");

function sh(cmd, cwd) { try { return execSync(cmd, { cwd, stdio: ["ignore", "pipe", "ignore"] }).toString().trim(); } catch (e) { return null; } }

function describeEnvironment(root) {
  const cpus = os.cpus() || [];
  return {
    capturedAt: new Date().toISOString(),
    platform: `${os.platform()} ${os.release()} (${os.arch()})`,
    cpuModel: cpus[0] ? cpus[0].model.trim() : "unknown",
    logicalCores: cpus.length,
    cpuMhzReported: cpus[0] ? cpus[0].speed : null,
    totalMemoryMB: Math.round(os.totalmem() / 1048576),
    node: process.version,
    v8: process.versions.v8,
    gitCommit: sh("git rev-parse --short HEAD", root),
    gitDirty: (sh("git status --porcelain", root) || "").split("\n").filter(Boolean).length > 0,
    isVirtualizedSandbox: null,   // not detectable reliably; the report author states it by hand
  };
}
module.exports = { describeEnvironment };
