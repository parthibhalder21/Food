"""Data poisoning gate: screens the latest sales upload before the model trains on it.

Data: real daily sales from Corporación Favorita (server/data/sales.csv, see fetch-data.yml).
The latest 7 days are the new upload; the 84 days before them are the reference history.

Two independent checks, either of which fails the pipeline:
  1. Z-score: each new daily figure is compared with that store's own history for the
     product family. |z| above the threshold means the figure is implausible. A second rule
     (the same one the app's data screen uses) also blocks figures more than 3x above or below
     the usual (median) level, which catches large drops that z-score misses on noisy lines.
  2. TensorFlow Data Validation (TFDV): the new week's statistics are validated against a
     schema learned from history (unknown stores or products, missing or negative values)
     and checked for distribution drift.

It also measures how well the z-score check catches poisoning: known fake figures are planted
in 12 real weeks and the check's flags are scored (accuracy, precision, recall, F1, error rate).

  python ml/data_gate.py                      check the latest week
  python ml/data_gate.py --inject-poison      demo: plant a poisoned figure first (the gate must fail)
"""
import argparse
import base64
import html
import io
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow_data_validation as tfdv  # noqa: E402
from tensorflow_data_validation.utils.display_util import get_statistics_html  # noqa: E402
from google.protobuf import text_format  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GREEN, RED, AMBER, GREY, INK = "#1f5a46", "#c4553a", "#c9922e", "#9aa5a0", "#1d2622"
SOURCE = "Corporación Favorita (Ecuador), Kaggle Store Sales: 10 largest stores, 8 perishable families"
ATTACKS = [("Spike x10", 10.0), ("Spike x3", 3.0), ("Drop to 10%", 0.1), ("Subtle x1.5", 1.5)]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=ROOT / "server/data/sales.csv")
    p.add_argument("--config", default=ROOT / "server/config/data-gates.json")
    p.add_argument("--out", default=ROOT / "artifacts/data-report")
    p.add_argument("--inject-poison", action="store_true", help="demo: plant a poisoned figure and an unknown product")
    return p.parse_args()


def split(d, end, cfg):
    """New upload = the 7 days ending at `end`; history = the 84 days before that."""
    new = d[(d.date > end - pd.Timedelta(days=cfg["newDays"])) & (d.date <= end)]
    hist = d[(d.date <= end - pd.Timedelta(days=cfg["newDays"])) & (d.date > end - pd.Timedelta(days=cfg["newDays"] + cfg["historyDays"]))]
    return hist, new


def zscores(hist, new, cfg):
    """Classic z-score per store and product family, plus the robust (median/MAD) version for reference."""
    g = hist.groupby(["site", "sku"])["units_sold"]
    ref = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=0), "median": g.median(),
                        "mad": g.apply(lambda s: (s - s.median()).abs().median()), "days": g.count()}).reset_index()
    z = new.merge(ref, on=["site", "sku"], how="left")
    z["z"] = (z["units_sold"] - z["mean"]) / z["std"].where(z["std"] > 0, 1)
    z["robust_z"] = 0.6745 * (z["units_sold"] - z["median"]) / z["mad"].where(z["mad"] > 0, 1)
    z["reason"] = ""
    z.loc[z["days"].isna() | (z["days"] < cfg["minHistoryDays"]), "reason"] = "no history for this store and product"
    r = cfg["ratioThreshold"]
    z["z_flag"] = z["z"].abs() > cfg["zScoreThreshold"]
    z["ratio_flag"] = (z["units_sold"] > r * z["median"]) | (z["units_sold"] < z["median"] / r)
    z.loc[(z["reason"] == "") & z["z_flag"], "reason"] = f"|z| above {cfg['zScoreThreshold']}"
    z.loc[(z["reason"] == "") & z["ratio_flag"], "reason"] = f"more than {r:g}x off the usual level"
    z["flagged"] = z["reason"] != ""
    return z


def scores(truth, pred):
    """Accuracy, precision, recall, F1 and error rate for yes/no decisions (positive = poisoned)."""
    truth, pred = np.asarray(truth, bool), np.asarray(pred, bool)
    tp, fp = int((truth & pred).sum()), int((~truth & pred).sum())
    fn, tn = int((truth & ~pred).sum()), int((~truth & ~pred).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": 100 * acc, "precision": 100 * prec,
            "recall": 100 * rec, "f1": 100 * f1, "errorRate": 100 * (1 - acc)}


def evaluate_detector(d, cfg, weeks=12, rate=0.10, seed=7):
    """Plant fake figures in 12 real weeks (10% of rows, four kinds of attack) and score the z-score check."""
    rng = np.random.default_rng(seed)
    end = d.date.max()
    rows = []
    for w in range(weeks):
        hist, new = split(d, end - pd.Timedelta(days=7 * w), cfg)
        new = new.copy()
        new["attack"] = ""
        idx = rng.choice(new.index, size=int(len(new) * rate), replace=False)
        for i, ix in enumerate(idx):
            name, factor = ATTACKS[i % len(ATTACKS)]
            new.loc[ix, "units_sold"] = round(new.loc[ix, "units_sold"] * factor, 1)
            new.loc[ix, "attack"] = name
        z = zscores(hist, new, cfg)
        z["week"] = w
        rows.append(z)
    r = pd.concat(rows, ignore_index=True)
    r["poisoned"] = r["attack"] != ""
    rules = {}
    for name, col in [("Z-score only", "z_flag"), ("3x rule only", "ratio_flag"), ("Z-score + 3x rule (the gate)", "flagged")]:
        m = scores(r["poisoned"], r[col])
        m["falseAlarmPct"] = float(100 * r.loc[~r.poisoned, col].mean())
        m["caught"] = {a: float(100 * r.loc[r.attack == a, col].mean()) for a, _ in ATTACKS}
        rules[name] = m
    overall = dict(rules["Z-score + 3x rule (the gate)"])
    overall["rows"], overall["poisonedRows"], overall["weeks"] = int(len(r)), int(r.poisoned.sum()), weeks
    return overall, rules


def tfdv_checks(hist, new, cfg, out):
    stats_hist = tfdv.generate_statistics_from_dataframe(hist)
    stats_new = tfdv.generate_statistics_from_dataframe(new)
    schema = tfdv.infer_schema(stats_hist)
    # Rules learned from history, tightened by hand: no negative sales, no missing values,
    # only known stores, products and categories, and a cap on distribution drift.
    tfdv.get_feature(schema, "units_sold").float_domain.min = 0
    tfdv.get_feature(schema, "onpromotion").int_domain.min = 0
    for name in ["date", "site", "sku", "category", "units_sold", "onpromotion", "holiday"]:
        tfdv.get_feature(schema, name).presence.min_fraction = 1.0
    tfdv.get_feature(schema, "date").ClearField("domain")  # new dates are expected every week
    tfdv.get_feature(schema, "site").drift_comparator.infinity_norm.threshold = cfg["categoryDriftLInf"]
    tfdv.get_feature(schema, "sku").drift_comparator.infinity_norm.threshold = cfg["categoryDriftLInf"]
    tfdv.get_feature(schema, "units_sold").drift_comparator.jensen_shannon_divergence.threshold = cfg["unitsDriftJensenShannon"]
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
    rows = {}
    for f in stats.datasets[0].features:
        name = f.path.step[0]
        if f.HasField("num_stats"):
            n = f.num_stats
            rows[name] = {"type": "number", "mean": n.mean, "std": n.std_dev, "min": n.min, "max": n.max,
                          "missing": n.common_stats.num_missing, "hist": n.histograms[0]}
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


def style(ax, grid="y"):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors="#4a5550", labelsize=8)
    ax.grid(axis=grid, color="#e6ebe8", linewidth=0.8)
    ax.set_axisbelow(True)


def charts(z, hist, new, fh, fn, cfg, det, by_attack):
    out, thr = {}, cfg["zScoreThreshold"]

    # 1. Z-score of every new figure, grouped by product family.
    fams = sorted(z["sku"].unique())
    pos = {f: i for i, f in enumerate(fams)}
    rng = np.random.default_rng(1)
    x = z["sku"].map(pos) + rng.uniform(-0.25, 0.25, len(z))
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axhspan(-thr, thr, color="#eef4f1", zorder=0)
    ax.axhline(thr, color=RED, ls="--", lw=1)
    ax.axhline(-thr, color=RED, ls="--", lw=1)
    ok = ~z["flagged"]
    ax.scatter(x[ok], z.loc[ok, "z"], s=12, color=GREEN, alpha=0.8, label="Within threshold")
    if (~ok).any():
        ax.scatter(x[~ok], z.loc[~ok, "z"].fillna(thr * 1.5).clip(upper=thr * 4), s=50, color=RED, marker="x", label="Blocked")
    ax.set_xticks(range(len(fams)), fams)
    ax.set_ylabel("z-score")
    ax.set_title(f"Z-score of every new daily figure, by product family (threshold ±{thr})", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    out["z"] = png(fig)

    # 2. The store-family line with the highest |z|, against its history.
    worst = z.loc[z["z"].abs().idxmax()]
    hh = hist[(hist.site == worst.site) & (hist.sku == worst.sku)]
    nn = new[(new.site == worst.site) & (new.sku == worst.sku)]
    m, s = hh["units_sold"].mean(), hh["units_sold"].std(ddof=0)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    alld = pd.concat([hh.date, nn.date])
    ax.fill_between(alld, m - thr * s, m + thr * s, color="#eef4f1", label=f"Mean ± {thr} std")
    ax.plot(hh.date, hh.units_sold, color=GREY, lw=1.2, label="History (84 days)")
    ax.plot(nn.date, nn.units_sold, color=GREEN, lw=1.8, marker="o", ms=3, label="Latest week")
    bad = nn.merge(z[z.flagged][["date", "site", "sku"]], on=["date", "site", "sku"])
    if len(bad):
        ax.scatter(bad.date, bad.units_sold, color=RED, s=60, marker="x", zorder=5, label="Blocked")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title(f"Highest |z| this week: {worst.site} · {worst.sku}", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    style(ax)
    out["series"] = png(fig)

    # 3. Units sold per day: TFDV histograms, history vs latest week.
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for stats, label, color, alpha in [(fh, "History (84 days)", GREY, 0.75), (fn, "Latest week", GREEN, 0.6)]:
        b = stats["units_sold"]["hist"].buckets
        total = sum(v.sample_count for v in b) or 1
        ax.bar([(v.low_value + v.high_value) / 2 for v in b], [v.sample_count / total for v in b],
               width=[(v.high_value - v.low_value) * 0.9 for v in b], color=color, alpha=alpha, label=label)
    ax.set_xlabel("Units sold per day")
    ax.set_ylabel("Share of rows")
    ax.set_title("Units sold per day (TFDV histogram)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    out["dist"] = png(fig)

    # 4. Average daily sales by category.
    h = hist.groupby("category")["units_sold"].mean()
    n = new.groupby("category")["units_sold"].mean().reindex(h.index)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    xs = np.arange(len(h))
    ax.bar(xs - 0.2, h.values, 0.4, color=GREY, label="History")
    ax.bar(xs + 0.2, n.values, 0.4, color=GREEN, label="Latest week")
    ax.set_xticks(xs, list(h.index), fontsize=8)
    ax.set_ylabel("Avg units per store per day")
    ax.set_title("Average daily sales by category", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    out["cat"] = png(fig)

    # 5. Detector: confusion matrix.
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    cm = np.array([[det["tn"], det["fp"]], [det["fn"], det["tp"]]])
    ax.imshow([[0, 1], [1, 0]], cmap=matplotlib.colors.ListedColormap(["#e3efe9", "#f6e1da"]))
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=14, color=INK, fontweight="bold")
    ax.set_xticks([0, 1], ["Passed", "Blocked"])
    ax.set_yticks([0, 1], ["Genuine", "Poisoned"])
    ax.set_xlabel("What the gate did")
    ax.set_ylabel("What the figure was")
    ax.set_title("Confusion matrix (12 real weeks)", fontsize=10, loc="left")
    ax.tick_params(length=0, labelsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    out["cm"] = png(fig)

    # 6. Detector: share of each attack caught, z-score alone vs the full gate.
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    names = [a for a, _ in ATTACKS]
    xs = np.arange(len(names))
    for off, (rule, color) in [(-0.2, ("Z-score only", GREY)), (0.2, ("Z-score + 3x rule (the gate)", GREEN))]:
        bars = ax.bar(xs + off, [by_attack[rule]["caught"][a] for a in names], 0.4, color=color, label=rule)
        for b in bars:
            ax.annotate(f"{b.get_height():.0f}%", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(xs, names, fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Caught (%)")
    ax.set_title("Share of each kind of attack caught", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style(ax)
    out["attacks"] = png(fig)
    return out


def tr(cells):
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def report(out, z, anomalies, fh, fn, imgs, cfg, passed, poisoned, det, by_attack, hist, new):
    flagged = z[z.flagged]
    feat_rows = []
    for name in ["units_sold", "onpromotion", "holiday", "site", "sku", "category"]:
        a, b = fh[name], fn[name]
        if a["type"] == "number":
            feat_rows.append(tr([name, "number", f"{a['mean']:.1f} / {b['mean']:.1f}", f"{a['std']:.1f} / {b['std']:.1f}",
                                 f"{a['min']:.0f}–{a['max']:.0f} / {b['min']:.0f}–{b['max']:.0f}", f"{a['missing']} / {b['missing']}"]))
        else:
            feat_rows.append(tr([name, "category", f"{a['unique']} / {b['unique']} unique", "", f"top: {html.escape(a['top'])}", f"{a['missing']} / {b['missing']}"]))
    status = ("<span class='pass'>PASSED</span> No figure is outside the z-score threshold and TFDV found no anomalies."
              if passed else "<span class='fail'>BLOCKED</span> Poisoned or invalid data found. The model will not train on it.")
    blocked_table = ("<div class='card'><table><tr><th>Date</th><th>Store</th><th>Product</th><th>Units</th><th>Usual (mean)</th><th>z</th><th>Robust z</th><th>Reason</th></tr>"
                     + "".join(tr([pd.Timestamp(r.date).date(), r.site, r.sku, r.units_sold, f"{r['mean']:.1f}" if pd.notna(r['mean']) else "-",
                                   f"{r.z:.1f}" if pd.notna(r.z) else "-", f"{r.robust_z:.1f}" if pd.notna(r.robust_z) else "-", r.reason]) for _, r in flagged.iterrows())
                     + "</table></div>") if len(flagged) else ""
    anomaly_html = ("<table><tr><th>Feature</th><th>Problem</th><th>Detail</th></tr>" + "".join(tr([html.escape(a), html.escape(b), html.escape(c)]) for a, b, c in anomalies) + "</table>"
                    if anomalies else "<p>None: every feature matches the schema learned from history, and drift is within limits.</p>")
    metric_rows = "".join(tr([k, f"{det[v]:.1f}%", d]) for k, v, d in [
        ("Accuracy", "accuracy", "share of all figures the gate judged correctly"),
        ("Precision", "precision", "of the figures it blocked, how many were really poisoned"),
        ("Recall", "recall", "of the poisoned figures, how many it blocked"),
        ("F1 score", "f1", "balance of precision and recall"),
        ("Error rate", "errorRate", "share of all figures judged wrongly (100 − accuracy)"),
        ("False alarm rate", "falseAlarmPct", "genuine figures blocked by mistake"),
    ])
    rule_rows = "".join(tr([f"<b>{k}</b>" if "gate" in k else k] + [f"{m[x]:.1f}%" for x in ["accuracy", "precision", "recall", "f1", "errorRate", "falseAlarmPct"]]) for k, m in by_attack.items())
    body = f"""<!doctype html><html><head><meta charset="utf-8"><title>FreshLoop data gate</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;color:{INK};background:#f5f6f4;margin:0}} main{{max-width:1080px;margin:0 auto;padding:32px 24px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 10px}} .sub{{color:#5b6762;margin:0 0 18px}}
.card{{background:#fff;border:1px solid #dde3df;border-radius:12px;padding:18px;margin:12px 0}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} .grid3{{display:grid;grid-template-columns:2fr 3fr;gap:12px}} img{{width:100%}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .kpi b{{display:block;font-size:26px}} .kpi span{{color:#5b6762;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border-bottom:1px solid #e6ebe8;padding:7px 8px;text-align:left}} th{{color:#5b6762;font-weight:600}}
.pass,.fail{{display:inline-block;padding:2px 10px;border-radius:99px;color:#fff;font-weight:700;margin-right:8px}} .pass{{background:{GREEN}}} .fail{{background:{RED}}} a{{color:{GREEN}}}
</style></head><body><main>
<h1>Data poisoning gate</h1>
<p class="sub">Real sales: {SOURCE}. Latest upload {pd.Timestamp(new.date.min()).date()} to {pd.Timestamp(new.date.max()).date()}, checked against the 84 days before it.{' Demo run: a poisoned figure was planted on purpose.' if poisoned else ''}</p>
<div class="card">{status}</div>
<div class="kpis">
<div class="card kpi"><b>{len(z)}</b><span>new daily figures checked</span></div>
<div class="card kpi"><b>{len(flagged)}</b><span>blocked (|z| &gt; {cfg['zScoreThreshold']} or {cfg['ratioThreshold']:g}x off)</span></div>
<div class="card kpi"><b>{len(anomalies)}</b><span>TFDV anomalies</span></div>
<div class="card kpi"><b>{z['z'].abs().max():.2f}</b><span>highest |z| this week</span></div>
</div>
<h2>Z-score check</h2>
<div class="card"><img src="data:image/png;base64,{imgs['z']}" alt="z-scores"></div>
<div class="card"><img src="data:image/png;base64,{imgs['series']}" alt="history"></div>
{blocked_table}
<h2>How well the gate catches poisoning</h2>
<p class="sub">Fake figures were planted in 10% of rows across the last {det['weeks']} real weeks ({det['poisonedRows']:,} of {det['rows']:,} figures): spikes of 10x and 3x, drops to 10%, and subtle 1.5x changes. The gate's decisions were scored against the truth.</p>
<div class="kpis">
<div class="card kpi"><b>{det['accuracy']:.1f}%</b><span>accuracy</span></div>
<div class="card kpi"><b>{det['precision']:.1f}%</b><span>precision</span></div>
<div class="card kpi"><b>{det['f1']:.1f}%</b><span>F1 score</span></div>
<div class="card kpi"><b>{det['errorRate']:.1f}%</b><span>error rate</span></div>
</div>
<div class="grid3"><div class="card"><img src="data:image/png;base64,{imgs['cm']}" alt="confusion matrix"></div>
<div class="card"><img src="data:image/png;base64,{imgs['attacks']}" alt="attacks caught"></div></div>
<div class="card"><table><tr><th>Rule</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Error rate</th><th>False alarms</th></tr>{rule_rows}</table></div>
<div class="card"><table><tr><th>Measure</th><th>Value</th><th>Meaning</th></tr>{metric_rows}</table>
<p class="sub" style="margin-top:12px">Z-score alone misses many large drops on lines whose sales swing a lot day to day; the 3x rule catches every one of them, so the gate uses both. Subtle 1.5x changes mostly get through: they sit inside normal day-to-day variation, so they show up later as drift (step 06) and in the waste gate instead.</p></div>
<h2>TensorFlow Data Validation</h2>
<div class="grid"><div class="card"><img src="data:image/png;base64,{imgs['dist']}" alt="distribution"></div>
<div class="card"><img src="data:image/png;base64,{imgs['cat']}" alt="categories"></div></div>
<div class="card"><table><tr><th>Feature</th><th>Type</th><th>Mean (history / new)</th><th>Std dev</th><th>Range</th><th>Missing</th></tr>{''.join(feat_rows)}</table></div>
<div class="card"><b>TFDV anomalies</b>{anomaly_html}
<p>Interactive view: <a href="tfdv-facets.html">TFDV Facets statistics</a> · Learned schema: <a href="schema.pbtxt">schema.pbtxt</a> · All scores: <a href="z-scores.csv">z-scores.csv</a> · <a href="../model-report/index.html">Forecast model report</a></p></div>
<p class="sub">Thresholds from server/config/data-gates.json: z-score {cfg['zScoreThreshold']}, category drift (L-infinity) {cfg['categoryDriftLInf']}, units drift (Jensen-Shannon) {cfg['unitsDriftJensenShannon']}, at least {cfg['minHistoryDays']} days of history.</p>
</main></body></html>"""
    (out / "index.html").write_text(body)


def main():
    a = args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.data, parse_dates=["date"])
    hist, new = split(d, d.date.max(), cfg)
    new = new.copy()
    if a.inject_poison:
        target = new[(new.sku == "DAIRY")].sort_values("date").iloc[-1]
        new.loc[target.name, "units_sold"] = round(target.units_sold * 10, 1)
        new = pd.concat([new, pd.DataFrame([{"date": target.date, "site": target.site, "sku": "DAIRY-FAKE", "category": "Dairy",
                                             "units_sold": 900.0, "onpromotion": 0, "holiday": 0}])], ignore_index=True)

    z = zscores(hist, new, cfg)
    z.assign(date=z.date.dt.date).to_csv(out / "z-scores.csv", index=False)
    as_text = lambda f: f.assign(date=f.date.dt.strftime("%Y-%m-%d")).reset_index(drop=True)
    stats_hist, stats_new, anomalies = tfdv_checks(as_text(hist), as_text(new), cfg, out)
    fh, fn = feature_stats(stats_hist), feature_stats(stats_new)
    det, by_attack = evaluate_detector(d, cfg)
    (out / "detector-metrics.json").write_text(json.dumps({"weeks": det["weeks"], "rows": det["rows"], "poisonedRows": det["poisonedRows"], "rules": by_attack}, indent=2, default=lambda v: round(v, 2)))
    passed = not z.flagged.any() and not anomalies
    report(out, z, anomalies, fh, fn, charts(z, hist, new, fh, fn, cfg, det, by_attack), cfg, passed, a.inject_poison, det, by_attack, hist, new)

    flagged = z[z.flagged]
    md = [f"### Data poisoning gate: {'PASSED' if passed else 'BLOCKED'}",
          f"Real Favorita sales. {len(z)} new daily figures checked against {len(hist)} rows of history. Highest |z| = {z['z'].abs().max():.2f} (limit {cfg['zScoreThreshold']}). TFDV anomalies: {len(anomalies)}.", ""]
    if len(flagged):
        md += ["| Date | Store | Product | Units | Usual | z | Reason |", "| --- | --- | --- | --- | --- | --- | --- |"]
        md += [f"| {pd.Timestamp(r.date).date()} | {r.site} | {r.sku} | {r.units_sold} | {r['mean']:.1f} | {r.z:.1f} | {r.reason.replace('|', chr(92) + '|')} |" if pd.notna(r['mean'])
               else f"| {pd.Timestamp(r.date).date()} | {r.site} | {r.sku} | {r.units_sold} | - | - | {r.reason} |" for _, r in flagged.iterrows()]
    if anomalies:
        md += ["", "| TFDV feature | Anomaly |", "| --- | --- |"] + [f"| {n} | {desc} |" for n, _, desc in anomalies]
    md += ["", f"**Poisoning detection, z-score + 3x rule** ({det['poisonedRows']} planted fakes in {det['rows']} real figures): accuracy {det['accuracy']:.1f}%, precision {det['precision']:.1f}%, recall {det['recall']:.1f}%, F1 {det['f1']:.1f}%, error rate {det['errorRate']:.1f}%.",
           "", "Full visual report: download the **data-report** artifact, or open Data quality in the app."]
    text = "\n".join(md)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
