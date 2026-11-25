#!/usr/bin/env node
// Model and data steps of the pipeline.
//
//   node scripts/model.js screen     03: block implausible sales data before training
//   node scripts/model.js train      02: retrain the sales forecast and tune the safety buffer
//   node scripts/model.js monitor    06: check forecast drift and list surplus to share
//
// Options: --data <csv>   --out <dir>   --model <json>
//          --sales-shift <n>  demo only: multiply actual sales by n, as if demand changed after release
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { loadPolicy, SERVER_ROOT } from "../src/config.js";
import { simulate } from "../src/sim.js";
import { openDb } from "../src/db.js";
import { seed } from "../src/seed.js";
import { assessBatch, dynamicPrice, toISODate } from "../src/risk.js";

const args = process.argv.slice(2);
const cmd = args[0];
const opt = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i > -1 && args[i + 1] !== undefined ? args[i + 1] : fallback;
};

const DATA = opt("data", path.join(SERVER_ROOT, "data", "weekly-sales.csv"));
const OUT = opt("out", path.join(SERVER_ROOT, "..", "artifacts"));
const SMOOTHING = 0.5; // weight of the latest week when updating a forecast
const SCREEN_RATIO = 3; // same rule as the data screen in the inventory API
const DRIFT_THRESHOLD_PCT = 20; // forecast error above this means the model has drifted
const MAX_STOCKOUT_PCT = 1.5; // the buffer search never trades more empty shelves than this

const sha256 = (file) => crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");

/** Append markdown to the GitHub Actions job summary, and print it. */
function report(md) {
  console.log(md);
  if (process.env.GITHUB_STEP_SUMMARY) fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, `${md}\n`);
}
function output(key, value) {
  if (process.env.GITHUB_OUTPUT) fs.appendFileSync(process.env.GITHUB_OUTPUT, `${key}=${value}\n`);
}
const table = (head, rows) => [`| ${head.join(" | ")} |`, `| ${head.map(() => "---").join(" | ")} |`, ...rows.map((r) => `| ${r.join(" | ")} |`)].join("\n");

const SHIFT = Number(opt("sales-shift", 1)) || 1;

function readSales(file) {
  const [header, ...lines] = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const cols = header.split(",");
  return lines.filter(Boolean).map((line, i) => {
    const v = line.split(",");
    const row = Object.fromEntries(cols.map((c, j) => [c, v[j]]));
    return { line: i + 2, site: row.site, sku: row.sku, forecast: Number(row.forecast_per_day), actual: +(Number(row.actual_per_day) * SHIFT).toFixed(1) };
  });
}

/** Weighted forecast error: total units missed / total units sold. */
function errorPct(rows, forecastOf) {
  const sold = rows.reduce((s, r) => s + r.actual, 0);
  const missed = rows.reduce((s, r) => s + Math.abs(r.actual - forecastOf(r)), 0);
  return sold ? +((100 * missed) / sold).toFixed(1) : 0;
}

function screen() {
  const rows = readSales(DATA);
  const problems = [];
  for (const r of rows) {
    if (!r.site || !r.sku) problems.push([r.line, r.site || "?", r.sku || "?", "missing site or product"]);
    else if (![r.forecast, r.actual].every((x) => Number.isFinite(x) && x >= 0)) problems.push([r.line, r.site, r.sku, "not a valid number"]);
    else if (r.forecast > 0 && (r.actual > r.forecast * SCREEN_RATIO || r.actual < r.forecast / SCREEN_RATIO)) {
      problems.push([r.line, r.site, r.sku, `sold ${r.actual}/day against a usual ${r.forecast}/day (more than ${SCREEN_RATIO}x off)`]);
    }
  }
  report(`### Data screen\n${rows.length} rows in \`${path.basename(DATA)}\`, sha256 \`${sha256(DATA).slice(0, 16)}\``);
  if (problems.length) {
    report(`\n**Blocked:** ${problems.length} implausible row(s). Fix or confirm them before the model is retrained.\n\n${table(["Line", "Site", "Product", "Problem"], problems)}`);
    process.exit(1);
  }
  report("\nPassed: every figure is within 3x of the site's usual sales.");
}

function train() {
  const rows = readSales(DATA);
  const policy = loadPolicy();

  // 1. Forecast model: exponential smoothing of each site's daily sales per product.
  const forecasts = rows.map((r) => ({ site: r.site, sku: r.sku, previous: r.forecast, forecast: +(SMOOTHING * r.actual + (1 - SMOOTHING) * r.forecast).toFixed(1) }));
  const before = errorPct(rows, (r) => r.forecast);

  // 2. Safety buffer: replay the waste simulation for each candidate buffer.
  const grid = [];
  for (let b = 0; b <= 0.5001; b += 0.05) {
    const res = simulate({ ...policy, safetyBuffer: +b.toFixed(2) });
    grid.push({ buffer: +b.toFixed(2), wastePct: res.wastePct, stockoutPct: res.stockoutPct });
  }
  const allowed = grid.filter((g) => g.stockoutPct <= MAX_STOCKOUT_PCT);
  const best = (allowed.length ? allowed : grid).reduce((a, g) => (g.wastePct < a.wastePct ? g : a));

  const model = {
    name: "freshloop-forecast",
    trainedAt: new Date().toISOString(),
    method: `exponential smoothing, alpha ${SMOOTHING}`,
    data: { file: path.basename(DATA), sha256: sha256(DATA), rows: rows.length },
    policy: { version: policy.version, sha256: sha256(path.join(SERVER_ROOT, "config", "risk-policy.json")), safetyBuffer: policy.safetyBuffer },
    forecastErrorBeforePct: before,
    recommendedSafetyBuffer: best.buffer,
    bufferSearch: grid,
    forecasts,
  };
  fs.mkdirSync(OUT, { recursive: true });
  const file = path.join(OUT, "forecast-model.json");
  fs.writeFileSync(file, `${JSON.stringify(model, null, 2)}\n`);

  report(`### Model trained\n${SHIFT !== 1 ? `Sales shifted by ${SHIFT}x for the demo. ` : ""}Retrained ${forecasts.length} forecasts on \`${model.data.file}\`. The forecasts in use were ${before}% off last week's sales.`);
  report(`\n${table(["Safety buffer", "Food wasted", "Stockouts"], grid.map((g) => [`${Math.round(g.buffer * 100)}%${g.buffer === policy.safetyBuffer ? " (in use)" : ""}${g.buffer === best.buffer ? " (best)" : ""}`, `${g.wastePct}%`, `${g.stockoutPct}%`]))}`);
  report(`\nBest buffer with stockouts at or under ${MAX_STOCKOUT_PCT}%: **${Math.round(best.buffer * 100)}%** (in use: ${Math.round(policy.safetyBuffer * 100)}%). Model saved as \`${path.relative(process.cwd(), file)}\`.`);
}

function monitor() {
  const policy = loadPolicy();
  const rows = readSales(DATA);

  // 1. Drift: are stores selling what the forecast said?
  const modelFile = opt("model", path.join(OUT, "forecast-model.json"));
  const model = fs.existsSync(modelFile) ? JSON.parse(fs.readFileSync(modelFile, "utf8")) : null;
  const lookup = new Map((model?.forecasts ?? []).map((f) => [`${f.site}/${f.sku}`, f.forecast]));
  const err = errorPct(rows, (r) => lookup.get(`${r.site}/${r.sku}`) ?? r.forecast);
  const drift = err > DRIFT_THRESHOLD_PCT;
  output("drift", drift);
  output("error_pct", err);
  report(`### Drift check\nForecast error against the latest sales: **${err}%** (limit ${DRIFT_THRESHOLD_PCT}%)${SHIFT !== 1 ? `, with sales shifted by ${SHIFT}x for the demo` : ""}.`);
  report(drift ? "\n**Drift detected:** the model is retrained and every check runs again." : "\nNo drift: the model in stores is still accurate.");

  // 2. Share surplus: what the network would move today, by the same engine the stores use.
  const today = toISODate();
  const db = openDb(":memory:");
  seed(db, { today });
  const batches = db.prepare(
    `SELECT o.name AS site, p.name AS product, p.base_price AS price, b.qty, b.daily_sales AS dailySales, b.expiry_date AS expiryDate
     FROM batches b JOIN orgs o ON o.id = b.org_id JOIN products p ON p.id = b.product_id`
  ).all();
  const share = batches
    .map((b) => ({ ...b, a: assessBatch(b, today, policy) }))
    .filter((b) => b.a.surplus > 0 && b.a.daysLeft >= 0)
    .sort((x, y) => x.a.daysLeft - y.a.daysLeft || y.a.surplus - x.a.surplus);
  const units = share.reduce((s, b) => s + b.a.surplus, 0);
  report(`\n### Surplus to share today\n${share.length} batches, ${units} units at risk of expiring unsold.\n`);
  report(table(["Site", "Product", "Units", "Days left", "Next step"], share.slice(0, 12).map((b) => {
    const step = b.a.daysLeft <= policy.ngoEscalationDaysLeft ? "Free to NGOs" : `Other store, then merchants at ${dynamicPrice(b.price, b.a.daysLeft, policy).discountPct}% off`;
    return [b.site.replace("FreshLoop ", ""), b.product, b.a.surplus, b.a.daysLeft, step];
  })));
}

const commands = { screen, train, monitor };
if (!commands[cmd]) {
  console.error("Usage: node scripts/model.js <screen|train|monitor> [--data file] [--out dir] [--model file] [--sales-shift n]");
  process.exit(2);
}
commands[cmd]();
