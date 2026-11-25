#!/usr/bin/env node
// DAST: attacks the running app over HTTP, the way an outsider would, and fails if any attack works.
// Runs in the pipeline against the signed release image, next to the OWASP ZAP baseline scan.
//
//   node scripts/dast.js http://localhost:4000
import fs from "node:fs";
import crypto from "node:crypto";

const BASE = (process.argv[2] || process.env.DAST_URL || "http://localhost:4000").replace(/\/$/, "");
const PASSWORD = "Demo@1234";
const results = [];

async function call(method, path, { token, body, raw, headers = {} } = {}) {
  const res = await fetch(BASE + path, {
    method,
    headers: { ...(body !== undefined || raw ? { "content-type": "application/json" } : {}), ...(token ? { authorization: `Bearer ${token}` } : {}), ...headers },
    body: raw ?? (body !== undefined ? JSON.stringify(body) : undefined),
  });
  const text = await res.text();
  return { status: res.status, headers: res.headers, text };
}

async function check(area, name, fn) {
  try {
    const detail = await fn();
    results.push({ area, name, ok: true, detail: detail || "" });
  } catch (err) {
    results.push({ area, name, ok: false, detail: err.message });
  }
}
const expect = (cond, msg) => { if (!cond) throw new Error(msg); };
const noLeak = (r) => expect(!/at\s+\S+\s+\(|node_modules|SqliteError|stack/i.test(r.text), `response leaks internals: ${r.text.slice(0, 120)}`);
const login = async (email) => {
  const r = await call("POST", "/api/auth/login", { body: { email, password: PASSWORD } });
  expect(r.status === 200, `login failed for ${email} (${r.status})`);
  return JSON.parse(r.text).token;
};
const b64url = (o) => Buffer.from(JSON.stringify(o)).toString("base64url");

async function main() {
  const health = await call("GET", "/api/health");
  if (health.status !== 200) throw new Error(`App is not reachable at ${BASE}`);
  const store = await login("koramangala@freshloop.test");
  const merchant = await login("lakshmi@kirana.test");

  // Security headers
  for (const path of ["/", "/api/health"]) {
    await check("Headers", `Security headers on ${path}`, async () => {
      const h = (await call("GET", path)).headers;
      expect(h.get("content-security-policy"), "no Content-Security-Policy");
      expect(h.get("x-content-type-options") === "nosniff", "no X-Content-Type-Options: nosniff");
      expect(h.get("x-frame-options") || /frame-ancestors/.test(h.get("content-security-policy")), "no clickjacking protection");
      expect(h.get("strict-transport-security"), "no Strict-Transport-Security");
      expect(!h.get("x-powered-by"), "X-Powered-By reveals the server");
      return "CSP, nosniff, frame protection, HSTS; no X-Powered-By";
    });
  }

  // Authentication
  for (const path of ["/api/inventory", "/api/orders", "/api/admin/audit", "/api/dashboard"]) {
    await check("Authentication", `No token: ${path}`, async () => {
      const r = await call("GET", path);
      expect(r.status === 401, `expected 401, got ${r.status}`);
      return "401";
    });
  }
  await check("Authentication", "Forged token (alg none)", async () => {
    const forged = `${b64url({ alg: "none", typ: "JWT" })}.${b64url({ sub: 1, role: "admin", orgId: 1 })}.`;
    const r = await call("GET", "/api/admin/audit", { token: forged });
    expect(r.status === 401, `expected 401, got ${r.status}`);
    return "401";
  });
  await check("Authentication", "Token signed with a guessed secret", async () => {
    const head = b64url({ alg: "HS256", typ: "JWT" });
    const body = b64url({ sub: 1, role: "admin", orgId: 1, exp: Math.floor(Date.now() / 1000) + 600 });
    const sig = crypto.createHmac("sha256", "secret").update(`${head}.${body}`).digest("base64url");
    const r = await call("GET", "/api/admin/audit", { token: `${head}.${body}.${sig}` });
    expect(r.status === 401, `expected 401, got ${r.status}`);
    return "401";
  });

  // Authorisation
  await check("Authorisation", "Merchant calls store inventory API", async () => {
    const r = await call("GET", "/api/inventory", { token: merchant });
    expect(r.status === 403, `expected 403, got ${r.status}`);
    return "403";
  });
  await check("Authorisation", "Store calls HQ admin API", async () => {
    const r = await call("GET", "/api/admin/orgs", { token: store });
    expect(r.status === 403, `expected 403, got ${r.status}`);
    return "403";
  });
  await check("Authorisation", "Store edits another store's batch (IDOR)", async () => {
    const own = JSON.parse((await call("GET", "/api/inventory", { token: store })).text).batches.map((b) => b.id);
    const other = Array.from({ length: 80 }, (_, i) => i + 1).find((id) => !own.includes(id));
    const r = await call("PATCH", `/api/inventory/batches/${other}`, { token: store, body: { qty: 0 } });
    expect([403, 404].includes(r.status), `expected 403/404, got ${r.status}`);
    return String(r.status);
  });
  await check("Authorisation", "Register as admin (mass assignment)", async () => {
    const r = await call("POST", "/api/auth/register", {
      body: { name: "Mallory", email: `mallory${Date.now()}@test.in`, password: "Password123", role: "admin", orgName: "Evil Co", address: "x", lat: 12.9, lng: 77.6, phone: "1" },
    });
    expect(r.status === 400, `expected 400, got ${r.status}`);
    return "400: role 'admin' rejected";
  });

  // Injection and input handling
  await check("Injection", "SQL injection in login", async () => {
    const r = await call("POST", "/api/auth/login", { body: { email: "' OR 1=1 --", password: "' OR '1'='1" } });
    expect([400, 401].includes(r.status), `expected 400/401, got ${r.status}`);
    noLeak(r);
    return String(r.status);
  });
  await check("Injection", "SQL injection in a URL id", async () => {
    const r = await call("PATCH", "/api/inventory/batches/1%20OR%201%3D1", { token: store, body: { qty: 1 } });
    expect([400, 403, 404].includes(r.status), `expected 400/403/404, got ${r.status}`);
    noLeak(r);
    return String(r.status);
  });
  await check("Injection", "Script tag in a product name is stored as text", async () => {
    const r = await call("POST", "/api/inventory/products", { token: store, body: { name: `<script>alert(1)</script> ${Date.now()}`, category: "Dairy", unit: "pack", basePrice: 10 } });
    expect([201, 400].includes(r.status), `unexpected ${r.status}`);
    expect((r.headers.get("content-type") || "").includes("application/json"), "API answered with something other than JSON");
    return `${r.status}, served as JSON (React escapes it on screen)`;
  });
  await check("Input", "Malformed JSON", async () => {
    const r = await call("POST", "/api/inventory/batches", { token: store, raw: "{not json" });
    expect(r.status === 400, `expected 400, got ${r.status}`);
    noLeak(r);
    return "400, no stack trace";
  });
  await check("Input", "Oversized request body (200 KB)", async () => {
    const r = await call("POST", "/api/auth/login", { raw: JSON.stringify({ email: "a@b.in", password: "x".repeat(200_000) }) });
    expect(r.status === 413, `expected 413, got ${r.status}`);
    return "413";
  });
  await check("Input", "Negative stock quantity", async () => {
    const r = await call("POST", "/api/inventory/batches", { token: store, body: { productId: 1, qty: -50, dailySales: 5, expiryDate: "2030-01-01" } });
    expect(r.status === 400, `expected 400, got ${r.status}`);
    return "400";
  });
  await check("Input", "Unknown API route", async () => {
    const r = await call("GET", "/api/does-not-exist", { token: store });
    expect(r.status === 404, `expected 404, got ${r.status}`);
    noLeak(r);
    return "404";
  });
  await check("Injection", "Path traversal to server files", async () => {
    for (const path of ["/%2e%2e/%2e%2e/etc/passwd", "/..%2f..%2f..%2fetc%2fpasswd", "/%2e%2e/server/config/risk-policy.json", "/%2e%2e/package.json"]) {
      const r = await call("GET", path);
      expect(!/root:x:0|"safetyBuffer"|"workspaces"/.test(r.text), `server file exposed at ${path}`);
    }
    return "no file outside the web folder served";
  });

  // Brute force (last: it locks this client out of sign-in for 15 minutes)
  await check("Brute force", "Password guessing is rate-limited", async () => {
    let limited = 0;
    for (let i = 0; i < 40 && !limited; i++) {
      const r = await call("POST", "/api/auth/login", { body: { email: "admin@freshloop.test", password: `guess${i}` } });
      if (r.status === 429) limited = i + 1;
    }
    expect(limited, "40 wrong passwords were all accepted for checking");
    return `blocked after ${limited} attempts`;
  });

  const failed = results.filter((r) => !r.ok);
  const md = [
    `### DAST: ${failed.length ? "FAILED" : "PASSED"} (${results.length - failed.length}/${results.length} attacks blocked)`,
    `Target: ${BASE}`,
    "",
    "| Area | Attack | Result | Detail |",
    "| --- | --- | --- | --- |",
    ...results.map((r) => `| ${r.area} | ${r.name} | ${r.ok ? "Blocked" : "**NOT BLOCKED**"} | ${r.detail.replace(/\|/g, "\\|")} |`),
  ].join("\n");
  console.log(md);
  if (process.env.GITHUB_STEP_SUMMARY) fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, `${md}\n`);
  process.exit(failed.length ? 1 : 0);
}

main().catch((err) => {
  console.error(`DAST could not run: ${err.message}`);
  process.exit(1);
});
