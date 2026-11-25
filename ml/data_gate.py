"""Data poisoning gate: screens the latest sales upload before the model trains on it.

Two independent checks, either of which fails the pipeline:
  1. Z-score: each new daily figure is compared with that site's own history for the
     product. |z| above the threshold means the figure is implausible (a typo or poisoning).
  2. TensorFlow Data Validation (TFDV): statistics of the new week are validated against a
     schema learned from history (unknown sites or products, missing or negative values)
     and checked for distribution drift.

It writes a visual report (data-report.html), the interactive TFDV Facets view
(tfdv-facets.html), the learned schema and every z-score to the output folder.

  python ml/data_gate.py                      check the data in server/data
  python ml/data_gate.py --inject-poison      demo: add a poisoned figure first (the gate must fail)
"""
import argparse
import base64
import html
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow_data_validation as tfdv  # noqa: E402
from tensorflow_data_validation.utils.display_util import get_statistics_html  # noqa: E402
from tensorflow_metadata.proto.v0 import schema_pb2  # noqa: E402
from google.protobuf import text_format  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GREEN, RED, AMBER, GREY = "#1f5a46", "#c4553a", "#c9922e", "#9aa5a0"


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--history", default=ROOT / "server/data/sales-history.csv")
    p.add_argument("--new", default=ROOT / "server/data/new-sales.csv")
    p.add_argument("--config", default=ROOT / "server/config/data-gates.json")
    p.add_argument("--out", default=ROOT / "artifacts/data-report")
    p.add_argument("--inject-poison", action="store_true", help="demo: add one poisoned figure and one unknown product")
    return p.parse_args()


def zscores(hist, new, cfg):
    """Classic z-score per site and product, plus the robust (median/MAD) version for reference."""
    g = hist.groupby(["site", "sku"])["units_sold"]
    ref = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=0), "median": g.median(),
                        "mad": g.apply(lambda s: (s - s.median()).abs().median()), "days": g.count()}).reset_index()
    z = new.merge(ref, on=["site", "sku"], how="left")
    z["z"] = (z["units_sold"] - z["mean"]) / z["std"].where(z["std"] > 0, 1)
    z["robust_z"] = 0.6745 * (z["units_sold"] - z["median"]) / z["mad"].where(z["mad"] > 0, 1)
    z["reason"] = ""
    z.loc[z["days"].isna() | (z["days"] < cfg["minHistoryDays"]), "reason"] = "no history for this site and product"
    z.loc[(z["reason"] == "") & (z["z"].abs() > cfg["zScoreThreshold"]), "reason"] = f"|z| above {cfg['zScoreThreshold']}"
    z["flagged"] = z["reason"] != ""
    return z


def tfdv_checks(hist, new, cfg, out):
    stats_hist = tfdv.generate_statistics_from_dataframe(hist)
    stats_new = tfdv.generate_statistics_from_dataframe(new)
    schema = tfdv.infer_schema(stats_hist)
    # Rules learned from history, tightened by hand: no negative sales, no missing values,
    # only known sites, products and categories, and a cap on distribution drift.
    units = tfdv.get_feature(schema, "units_sold")
    units.int_domain.min = 0
    for name in ["date", "site", "sku", "category", "units_sold", "price"]:
        tfdv.get_feature(schema, name).presence.min_fraction = 1.0
    tfdv.get_feature(schema, "date").ClearField("domain")  # new dates are expected every week
    tfdv.get_feature(schema, "site").drift_comparator.infinity_norm.threshold = cfg["categoryDriftLInf"]
    tfdv.get_feature(schema, "sku").drift_comparator.infinity_norm.threshold = cfg["categoryDriftLInf"]
    units.drift_comparator.jensen_shannon_divergence.threshold = cfg["unitsDriftJensenShannon"]
    anomalies = tfdv.validate_statistics(stats_new, schema, previous_statistics=stats_hist)

    (out / "schema.pbtxt").write_text(text_format.MessageToString(schema))
    facets = get_statistics_html(lhs_statistics=stats_new, rhs_statistics=stats_hist, lhs_name="Latest week", rhs_name="History (84 days)")
    (out / "tfdv-facets.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>TFDV statistics: latest week vs history</title>"
        "<body style='font-family:sans-serif;margin:24px'><h2>TFDV statistics: latest week vs history</h2>"
        "<p>Interactive Facets view from TensorFlow Data Validation. Needs an internet connection.</p>" + facets + "</body>")
    found = [(name, a.short_description, a.description) for name, a in anomalies.anomaly_info.items()]
    return stats_hist, stats_new, found


def feature_stats(stats):
    """Per-feature numbers from a TFDV statistics proto."""
    rows = {}
    for f in stats.datasets[0].features:
        name = f.path.step[0]
        if f.HasField("num_stats"):
            n = f.num_stats
            rows[name] = {"type": "number", "mean": n.mean, "std": n.std_dev, "min": n.min, "max": n.max,
                          "zeros": n.num_zeros, "missing": n.common_stats.num_missing, "hist": n.histograms[0]}
        else:
            s = f.string_stats
            rows[name] = {"type": "category", "unique": s.unique, "top": s.top_values[0].value if s.top_values else "",
                          "missing": s.common_stats.num_missing}
    return rows


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def style(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors="#4a5550", labelsize=8)
    ax.grid(axis="y", color="#e6ebe8", linewidth=0.8)
    ax.set_axisbelow(True)


def charts(z, hist, new, fh, fn, cfg):
    out = {}
    thr = cfg["zScoreThreshold"]

    # 1. Z-score of every new figure, by product line.
    z = z.copy()
    z["item"] = z["site"].str[:3].str.title() + " " + z["sku"]
    order = {k: i for i, k in enumerate(sorted(z["item"].unique()))}
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ok, bad = z[~z["flagged"]], z[z["flagged"]]
    ax.axhspan(-thr, thr, color="#eef4f1", zorder=0)
    ax.axhline(thr, color=RED, ls="--", lw=1)
    ax.axhline(-thr, color=RED, ls="--", lw=1)
    ax.scatter(ok["item"].map(order), ok["z"], s=14, color=GREEN, label="Within threshold")
    if len(bad):
        ax.scatter(bad["item"].map(order), bad["z"].fillna(thr * 1.5), s=46, color=RED, marker="x", label="Blocked")
    ax.set_xticks(range(len(order)), list(order), rotation=90)
    ax.set_ylabel("z-score")
    ax.set_title(f"Z-score of each new daily figure against its own history (threshold ±{thr})", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    out["z"] = png(fig)

    # 2. Distribution of units sold: TFDV histograms, history vs latest week.
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for stats, label, color in [(fh, "History (84 days)", GREY), (fn, "Latest week", GREEN)]:
        b = stats["units_sold"]["hist"].buckets
        total = sum(x.sample_count for x in b) or 1
        ax.bar([(x.low_value + x.high_value) / 2 for x in b], [x.sample_count / total for x in b],
               width=[(x.high_value - x.low_value) * 0.9 for x in b], color=color, alpha=0.75 if color == GREY else 0.6, label=label)
    ax.set_xlabel("Units sold per day")
    ax.set_ylabel("Share of rows")
    ax.set_title("Units sold per day (TFDV histogram)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    out["dist"] = png(fig)

    # 3. Average daily sales by category, history vs latest week.
    h = hist.groupby("category")["units_sold"].mean()
    n = new.groupby("category")["units_sold"].mean().reindex(h.index)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = range(len(h))
    ax.bar([i - 0.2 for i in x], h.values, 0.4, color=GREY, label="History")
    ax.bar([i + 0.2 for i in x], n.values, 0.4, color=GREEN, label="Latest week")
    ax.set_xticks(list(x), list(h.index), fontsize=8)
    ax.set_ylabel("Avg units per day")
    ax.set_title("Average daily sales by category", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    out["cat"] = png(fig)

    # 4. One product's history with the new week, to show what "normal" looks like.
    worst = z.loc[z["z"].abs().idxmax()]
    key = (worst["site"], worst["sku"])
    hh = hist[(hist.site == key[0]) & (hist.sku == key[1])]
    nn = new[(new.site == key[0]) & (new.sku == key[1])]
    m, s = hh["units_sold"].mean(), hh["units_sold"].std(ddof=0)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.fill_between(pd.to_datetime(pd.concat([hh.date, nn.date])), m - thr * s, m + thr * s, color="#eef4f1", label=f"Mean ± {thr} std")
    ax.plot(pd.to_datetime(hh.date), hh.units_sold, color=GREY, lw=1.2, label="History")
    ax.plot(pd.to_datetime(nn.date), nn.units_sold, color=GREEN, lw=1.8, marker="o", ms=3, label="Latest week")
    flagged = nn.merge(z[z.flagged][["date", "site", "sku"]], on=["date", "site", "sku"])
    if len(flagged):
        ax.scatter(pd.to_datetime(flagged.date), flagged.units_sold, color=RED, s=60, marker="x", zorder=5, label="Blocked")
    ax.set_title(f"Highest z-score: {key[0].title()} · {key[1]}", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    style(ax)
    out["series"] = png(fig)
    return out


def report(out, z, anomalies, fh, fn, imgs, cfg, passed, poisoned):
    def tr(cells):
        return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

    flagged = z[z.flagged]
    feat_rows = []
    for name in ["units_sold", "price", "site", "sku", "category", "date"]:
        a, b = fh[name], fn[name]
        if a["type"] == "number":
            feat_rows.append(tr([name, "number", f"{a['mean']:.1f} / {b['mean']:.1f}", f"{a['std']:.1f} / {b['std']:.1f}",
                                 f"{a['min']:.0f}–{a['max']:.0f} / {b['min']:.0f}–{b['max']:.0f}", f"{a['missing']} / {b['missing']}"]))
        else:
            feat_rows.append(tr([name, "category", f"{a['unique']} / {b['unique']} unique", "", f"top: {html.escape(a['top'])}", f"{a['missing']} / {b['missing']}"]))
    status = ("<span class='pass'>PASSED</span> No figure is outside the z-score threshold and TFDV found no anomalies."
              if passed else "<span class='fail'>BLOCKED</span> Poisoned or invalid data found. The model will not train on it.")
    body = f"""<!doctype html><html><head><meta charset="utf-8"><title>FreshLoop data gate</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;color:#1d2622;background:#f5f6f4;margin:0}}
main{{max-width:1080px;margin:0 auto;padding:32px 24px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 10px}}
.sub{{color:#5b6762;margin:0 0 18px}} .card{{background:#fff;border:1px solid #dde3df;border-radius:12px;padding:18px;margin:12px 0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} img{{width:100%}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .kpi b{{display:block;font-size:26px}} .kpi span{{color:#5b6762;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border-bottom:1px solid #e6ebe8;padding:7px 8px;text-align:left}} th{{color:#5b6762;font-weight:600}}
.pass,.fail{{display:inline-block;padding:2px 10px;border-radius:99px;color:#fff;font-weight:700;margin-right:8px}} .pass{{background:{GREEN}}} .fail{{background:{RED}}}
a{{color:{GREEN}}}
</style></head><body><main>
<h1>Data poisoning gate</h1>
<p class="sub">Latest sales upload checked against 84 days of history before the model trains on it.{' Demo run: a poisoned figure was injected on purpose.' if poisoned else ''}</p>
<div class="card">{status}</div>
<div class="kpis">
<div class="card kpi"><b>{len(z)}</b><span>new daily figures checked</span></div>
<div class="card kpi"><b>{len(flagged)}</b><span>blocked by z-score (|z| &gt; {cfg['zScoreThreshold']})</span></div>
<div class="card kpi"><b>{len(anomalies)}</b><span>TFDV anomalies</span></div>
<div class="card kpi"><b>{z['z'].abs().max():.2f}</b><span>highest |z| this week</span></div>
</div>
<h2>Z-score check</h2>
<div class="card"><img src="data:image/png;base64,{imgs['z']}" alt="z-scores"></div>
<div class="card"><img src="data:image/png;base64,{imgs['series']}" alt="history"></div>
{"<div class='card'><table><tr><th>Date</th><th>Site</th><th>Product</th><th>Units</th><th>Usual (mean)</th><th>z</th><th>Robust z</th><th>Reason</th></tr>" + "".join(tr([r.date, r.site, r.sku, r.units_sold, f"{r['mean']:.1f}" if pd.notna(r['mean']) else "-", f"{r.z:.1f}" if pd.notna(r.z) else "-", f"{r.robust_z:.1f}" if pd.notna(r.robust_z) else "-", r.reason]) for _, r in flagged.iterrows()) + "</table></div>" if len(flagged) else ""}
<h2>TensorFlow Data Validation</h2>
<div class="grid"><div class="card"><img src="data:image/png;base64,{imgs['dist']}" alt="distribution"></div>
<div class="card"><img src="data:image/png;base64,{imgs['cat']}" alt="categories"></div></div>
<div class="card"><table><tr><th>Feature</th><th>Type</th><th>Mean (history / new)</th><th>Std dev</th><th>Range</th><th>Missing</th></tr>{''.join(feat_rows)}</table></div>
<div class="card"><b>TFDV anomalies</b>{"<table><tr><th>Feature</th><th>Problem</th><th>Detail</th></tr>" + "".join(tr([html.escape(a), html.escape(b), html.escape(c)]) for a, b, c in anomalies) + "</table>" if anomalies else "<p>None: every feature matches the schema learned from history, and drift is within limits.</p>"}
<p>Interactive view: <a href="tfdv-facets.html">TFDV Facets statistics</a> · Learned schema: <a href="schema.pbtxt">schema.pbtxt</a> · All scores: <a href="z-scores.csv">z-scores.csv</a></p></div>
<p class="sub">Thresholds from server/config/data-gates.json: z-score {cfg['zScoreThreshold']}, category drift (L-infinity) {cfg['categoryDriftLInf']}, units drift (Jensen-Shannon) {cfg['unitsDriftJensenShannon']}, at least {cfg['minHistoryDays']} days of history.</p>
</main></body></html>"""
    (out / "index.html").write_text(body)


def main():
    a = args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hist = pd.read_csv(a.history)
    new = pd.read_csv(a.new)
    if a.inject_poison:
        last = new["date"].max()
        new = pd.concat([new, pd.DataFrame([
            {"date": last, "site": "koramangala", "sku": "MILK-TND-500", "category": "Dairy", "units_sold": 400, "price": 30},
            {"date": last, "site": "koramangala", "sku": "MILK-FAKE-1L", "category": "Dairy", "units_sold": 90, "price": 30},
        ])], ignore_index=True)

    z = zscores(hist, new, cfg)
    z.to_csv(out / "z-scores.csv", index=False)
    stats_hist, stats_new, anomalies = tfdv_checks(hist, new, cfg, out)
    fh, fn = feature_stats(stats_hist), feature_stats(stats_new)
    passed = not z.flagged.any() and not anomalies
    report(out, z, anomalies, fh, fn, charts(z, hist, new, fh, fn, cfg), cfg, passed, a.inject_poison)

    flagged = z[z.flagged]
    md = [f"### Data poisoning gate: {'PASSED' if passed else 'BLOCKED'}",
          f"{len(z)} new daily figures checked against {len(hist)} rows of history. Highest |z| = {z['z'].abs().max():.2f} (limit {cfg['zScoreThreshold']}). TFDV anomalies: {len(anomalies)}.", ""]
    if len(flagged):
        md += ["| Date | Site | Product | Units | Usual | z | Reason |", "| --- | --- | --- | --- | --- | --- | --- |"]
        md += [f"| {r.date} | {r.site} | {r.sku} | {r.units_sold} | {r['mean']:.1f} | {r.z:.1f} | {r.reason.replace('|', chr(92) + '|')} |" if pd.notna(r['mean'])
               else f"| {r.date} | {r.site} | {r.sku} | {r.units_sold} | - | - | {r.reason} |" for _, r in flagged.iterrows()]
    if anomalies:
        md += ["", "| TFDV feature | Anomaly |", "| --- | --- |"] + [f"| {n} | {d} |" for n, _, d in anomalies]
    md += ["", "Full visual report: download the **data-report** artifact and open `index.html`."]
    text = "\n".join(md)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
