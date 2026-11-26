"""Trains the LightGBM demand forecast on real sales and decides whether it replaces the current model.

Data: real daily sales from Corporación Favorita (server/data/sales.csv, see fetch-data.yml).
The last 28 days are held out as the test period; everything before is training data.

Champion:   exponential smoothing (the model in stores today).
Challenger: LightGBM with quantile loss. P50 is the most likely sale; P85 is the stock level we are
            85% sure covers demand, so each product family gets its own safety margin.

Two kinds of measurement:
  Forecasting (a number):  MAE, RMSE, error rate (WAPE) and accuracy (100 - WAPE).
  Surplus-risk flag (yes/no): each morning a store holds its usual order (same day last week + 15%).
      The system flags lines where more than 10% of that stock will be left unsold, so they can be
      listed for merchants early. Flags are scored against what actually sold: accuracy,
      precision, recall, F1 and error rate.

The challenger ships only if it beats the champion on both forecast error and surplus-flag F1.

  python ml/train_model.py --out artifacts/model-report
"""
import argparse
import base64
import io
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GREEN, CLAY, AMBER, GREY, INK = "#1f5a46", "#c4553a", "#c9922e", "#9aa5a0", "#1d2622"
SOURCE = "Corporación Favorita (Ecuador), Kaggle Store Sales: 10 largest stores, 8 perishable families"
FIXED_BUFFER = 0.15  # the current policy (server/config/risk-policy.json)
SURPLUS_SHARE = 0.10  # a line is "at risk" if more than 10% of its stock would be left unsold
SEED = 42
FEATURES = ["lag_1", "lag_2", "lag_7", "lag_14", "mean_7", "std_7", "level", "dow", "dom", "payday",
            "onpromotion", "promo_ratio", "holiday", "site", "sku", "category"]
FEATURE_NAMES = {
    "lag_1": "Sales yesterday", "lag_2": "Sales 2 days ago", "lag_7": "Same day last week", "lag_14": "Same day 2 weeks ago",
    "mean_7": "7-day average", "std_7": "7-day variability", "level": "Store-family size", "dow": "Day of week",
    "dom": "Day of month", "payday": "Payday (15th, month end)", "onpromotion": "Items on promotion",
    "promo_ratio": "Promotion vs usual", "holiday": "National holiday", "site": "Store", "sku": "Product family", "category": "Category",
}
MODELS = ["7-day average", "Same day last week", "Exponential smoothing (current)", "LightGBM (P50)"]
COLS = {"7-day average": "f_mean7", "Same day last week": "f_lag7", "Exponential smoothing (current)": "f_exp", "LightGBM (P50)": "f_p50"}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=ROOT / "server/data/sales.csv")
    p.add_argument("--test-days", type=int, default=28)
    p.add_argument("--out", default=ROOT / "artifacts/model-report")
    return p.parse_args()


def build_features(d, test_days):
    d = d.sort_values(["site", "sku", "date"]).reset_index(drop=True)
    g = d.groupby(["site", "sku"])
    past = g["units_sold"].shift(1)  # only days before the one being forecast
    roll = lambda s, w, f: s.groupby([d.site, d.sku]).transform(lambda x: getattr(x.rolling(w), f)())  # noqa: E731
    d["mean_28"] = roll(past, 28, "mean")
    base = d["mean_28"] + 1  # every sales feature is relative to the line's own level, so one model fits all sizes
    for k in [1, 2, 7, 14]:
        d[f"lag_{k}"] = g["units_sold"].shift(k) / base
    d["mean_7"] = roll(past, 7, "mean") / base
    d["std_7"] = roll(past, 7, "std") / base
    d["level"] = np.log1p(d["mean_28"])
    d["exp_smooth"] = past.groupby([d.site, d.sku]).transform(lambda s: s.ewm(alpha=0.5, adjust=False).mean())
    promo_28 = roll(g["onpromotion"].shift(1), 28, "mean")
    d["promo_ratio"] = d["onpromotion"] / (promo_28 + 1)
    d["dow"] = d.date.dt.dayofweek
    d["dom"] = d.date.dt.day
    d["payday"] = ((d.dom == 15) | d.date.dt.is_month_end).astype(int)  # public-sector wages in Ecuador
    for c in ["site", "sku", "category"]:
        d[c] = d[c].astype("category")
    d["target"] = d["units_sold"] / base
    d = d.dropna(subset=["mean_28", "lag_14", "std_7"]).reset_index(drop=True)
    d["split"] = np.where(d.date > d.date.max() - pd.Timedelta(days=test_days), "test", "train")
    return d


def train(train_df, alpha):
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=alpha, n_estimators=600, learning_rate=0.03, num_leaves=31,
        min_child_samples=20, subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
        random_state=SEED, deterministic=True, force_col_wise=True, verbose=-1,
    )
    model.fit(train_df[FEATURES], train_df["target"], categorical_feature=["site", "sku", "category"])
    return model


def regression(actual, forecast):
    err = forecast - actual
    wape = 100 * np.abs(err).sum() / actual.sum()
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())), "errorRate": float(wape),
            "accuracy": float(100 - wape), "biasPct": float(100 * err.sum() / actual.sum())}


def classification(truth, pred):
    truth, pred = np.asarray(truth, bool), np.asarray(pred, bool)
    tp, fp = int((truth & pred).sum()), int((~truth & pred).sum())
    fn, tn = int((truth & ~pred).sum()), int((~truth & ~pred).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": 100 * acc, "precision": 100 * prec,
            "recall": 100 * rec, "f1": 100 * f1, "errorRate": 100 * (1 - acc)}


def stock_check(actual, keep):
    over, short = np.maximum(0, keep - actual).sum(), np.maximum(0, actual - keep).sum()
    return float(100 * over / actual.sum()), float(100 * short / actual.sum())


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


def label_bars(ax, bars, fmt="{:.1f}%", size=8):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=size, color=INK, xytext=(0, 2), textcoords="offset points")


def tr(cells, bold=False):
    return "<tr>" + "".join(f"<td>{'<b>' if bold else ''}{c}{'</b>' if bold else ''}</td>" for c in cells) + "</tr>"


def main():
    a = args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(a.data, parse_dates=["date"])
    df = build_features(raw, a.test_days)
    tr_df, te = df[df.split == "train"].copy(), df[df.split == "test"].copy()

    p50, p85 = train(tr_df, 0.5), train(tr_df, 0.85)
    base = te["mean_28"] + 1
    te["f_p50"] = (p50.predict(te[FEATURES]) * base).clip(lower=0)
    te["f_p85"] = np.maximum(p85.predict(te[FEATURES]) * base, te["f_p50"])
    te["f_mean7"] = te["mean_7"] * base
    te["f_lag7"] = te["lag_7"] * base
    te["f_exp"] = te["exp_smooth"]
    y = te["units_sold"].to_numpy(float)

    # Forecasting metrics
    reg = {m: regression(y, te[COLS[m]].to_numpy(float)) for m in MODELS}
    fam = te.groupby("sku", observed=True).apply(
        lambda g: pd.Series({"LightGBM": 100 * np.abs(g.f_p50 - g.units_sold).sum() / g.units_sold.sum(),
                             "Current": 100 * np.abs(g.f_exp - g.units_sold).sum() / g.units_sold.sum()}), include_groups=False)

    # Surplus-risk flags: stock = usual order (same day last week + 15%); at risk = >10% of it left unsold.
    stock = te["f_lag7"] * (1 + FIXED_BUFFER)
    limit = (1 - SURPLUS_SHARE) * stock
    truth = te["units_sold"] < limit
    cls = {m: classification(truth, te[COLS[m]] < limit) for m in MODELS}

    # Stock each policy would hold
    policies = {
        "Current: exp. smoothing + 15% buffer": te["f_exp"] * (1 + FIXED_BUFFER),
        "LightGBM P50 + 15% buffer": te["f_p50"] * (1 + FIXED_BUFFER),
        "LightGBM P85 (learned buffer)": te["f_p85"],
    }
    stock_pol = {k: stock_check(y, v.to_numpy(float)) for k, v in policies.items()}
    coverage = float(100 * (y <= te["f_p85"]).mean())
    te["buffer_pct"] = 100 * (te["f_p85"] / te["f_p50"].clip(lower=0.1) - 1)
    buffers = te.groupby("sku", observed=True)["buffer_pct"].median().sort_values()

    champ, chall = "Exponential smoothing (current)", "LightGBM (P50)"
    wins_acc = reg[chall]["errorRate"] < reg[champ]["errorRate"]
    wins_f1 = cls[chall]["f1"] > cls[champ]["f1"]
    selected = "lightgbm-quantile" if (wins_acc and wins_f1) else "exp-smoothing"

    p50.booster_.save_model(str(out / "lightgbm-p50.txt"))
    p85.booster_.save_model(str(out / "lightgbm-p85.txt"))
    te.assign(date=te.date.dt.date)[["date", "site", "sku", "units_sold", "f_exp", "f_p50", "f_p85"]].round(1).to_csv(out / "forecasts.csv", index=False)
    r2 = lambda d: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}  # noqa: E731
    metrics = {
        "data": SOURCE,
        "trainedOn": {"rows": int(len(tr_df)), "from": str(tr_df.date.min().date()), "to": str(tr_df.date.max().date())},
        "testedOn": {"rows": int(len(te)), "from": str(te.date.min().date()), "to": str(te.date.max().date())},
        "forecasting": {m: r2(v) for m, v in reg.items()},
        "surplusRiskFlags": {m: r2(v) for m, v in cls.items()},
        "stockCheckPct": {k: {"overStocked": round(o, 2), "short": round(s, 2)} for k, (o, s) in stock_pol.items()},
        "p85CoveragePct": round(coverage, 1),
        "learnedBufferPct": {k: round(v, 1) for k, v in buffers.items()},
        "selected": selected,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---------- charts
    imgs = {}
    colors = [GREY, GREY, AMBER, GREEN]
    short_names = [m.replace(" (", "\n(") for m in MODELS]

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    bars = ax.bar(range(4), [reg[m]["errorRate"] for m in MODELS], color=colors, width=0.6)
    label_bars(ax, bars)
    ax.set_xticks(range(4), short_names, fontsize=8)
    ax.set_ylabel("Error rate (WAPE %)")
    ax.set_title("Forecast error rate on the 28 test days (lower is better)", fontsize=10, loc="left")
    style(ax)
    imgs["acc"] = png(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    keys = ["accuracy", "precision", "recall", "f1"]
    xs = np.arange(len(keys))
    for i, (m, c) in enumerate([(champ, AMBER), (chall, GREEN)]):
        bars = ax.bar(xs + (i - 0.5) * 0.36, [cls[m][k] for k in keys], 0.36, color=c, label=m)
        label_bars(ax, bars, "{:.0f}%", 7)
    ax.set_xticks(xs, ["Accuracy", "Precision", "Recall", "F1 score"], fontsize=8)
    ax.set_ylim(0, 128)
    ax.set_title("Surplus-risk flags: current model vs LightGBM", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    style(ax)
    imgs["cls"] = png(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    c = cls[chall]
    cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    ax.imshow([[0, 1], [1, 0]], cmap=matplotlib.colors.ListedColormap(["#e3efe9", "#f6e1da"]))
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=14, color=INK, fontweight="bold")
    ax.set_xticks([0, 1], ["Not flagged", "Flagged"])
    ax.set_yticks([0, 1], ["Sold through", "Surplus left"])
    ax.set_xlabel("LightGBM's flag")
    ax.set_ylabel("What actually happened")
    ax.set_title("Confusion matrix (LightGBM)", fontsize=10, loc="left")
    ax.tick_params(length=0, labelsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    imgs["cm"] = png(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    pn = list(stock_pol)
    xs = np.arange(len(pn))
    b1 = ax.bar(xs - 0.18, [stock_pol[p][0] for p in pn], 0.36, color=AMBER, label="Over-stocked (risk of waste)")
    b2 = ax.bar(xs + 0.18, [stock_pol[p][1] for p in pn], 0.36, color=CLAY, label="Short (stockouts)")
    label_bars(ax, list(b1) + list(b2))
    ax.set_xticks(xs, [p.replace(": ", ":\n").replace(" + ", "\n+ ").replace(" (", "\n(") for p in pn], fontsize=8)
    ax.set_ylabel("% of demand")
    ax.set_title("Stock level each policy would hold", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    imgs["stock"] = png(fig)

    key = te.groupby(["site", "sku"], observed=True).units_sold.sum().idxmax()
    key = (key[0], "DAIRY")
    s_all = df[(df.site == key[0]) & (df.sku == key[1])].tail(70)
    s_te = te[(te.site == key[0]) & (te.sku == key[1])]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(s_all.date, s_all.units_sold, color=GREY, lw=1.3, marker="o", ms=2, label="Actual sales")
    ax.fill_between(s_te.date, s_te.f_p50, s_te.f_p85, color=GREEN, alpha=0.15, label="LightGBM P50 to P85")
    ax.plot(s_te.date, s_te.f_p50, color=GREEN, lw=2, label="LightGBM forecast (P50)")
    ax.plot(s_te.date, s_te.f_exp, color=AMBER, lw=1.5, ls="--", label="Exponential smoothing")
    ax.axvline(mdates.date2num(pd.Timestamp(s_te.date.min()).to_pydatetime()) - 0.5, color=GREY, ls=":", lw=1)
    ax.text(pd.Timestamp(s_te.date.min()), ax.get_ylim()[1], " test period", va="top", fontsize=8, color="#4a5550")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title(f"{key[0]} · Dairy: forecasts against real sales", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower left")
    style(ax)
    imgs["series"] = png(fig)

    imp = pd.Series(p50.booster_.feature_importance("gain"), index=FEATURES)
    imp = (100 * imp / imp.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.barh([FEATURE_NAMES[f] for f in imp.index], imp.values, color=GREEN)
    ax.set_xlabel("Share of the model's decisions (%)")
    ax.set_title("What the model relies on", fontsize=10, loc="left")
    style(ax, grid="x")
    imgs["imp"] = png(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.barh(buffers.index, buffers.values, color=GREEN)
    ax.axvline(100 * FIXED_BUFFER, color=CLAY, ls="--", lw=1.2, label="Fixed 15% buffer today")
    ax.set_xlim(0, max(100 * FIXED_BUFFER, buffers.max()) * 1.25)
    ax.set_xlabel("Safety margin above the likely sale (%)")
    ax.set_title("Learned safety margin per family (P85 vs P50)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax, grid="x")
    imgs["buf"] = png(fig)

    # ---------- report
    ship = selected == "lightgbm-quantile"
    rc, rl = reg[champ], reg[chall]
    cc, cl = cls[champ], cls[chall]
    verdict = (f"<span class='pass'>LIGHTGBM SELECTED</span> Lower error rate ({rl['errorRate']:.1f}% vs {rc['errorRate']:.1f}%) and better surplus flags (F1 {cl['f1']:.1f}% vs {cc['f1']:.1f}%)."
               if ship else "<span class='keep'>CURRENT MODEL KEPT</span> LightGBM did not beat exponential smoothing on both error rate and surplus-flag F1, so it does not ship.")
    reg_rows = "".join(tr([m, f"{reg[m]['mae']:.1f}", f"{reg[m]['rmse']:.1f}", f"{reg[m]['errorRate']:.1f}%", f"{reg[m]['accuracy']:.1f}%", f"{reg[m]['biasPct']:+.1f}%"], m == chall) for m in MODELS)
    cls_rows = "".join(tr([m, f"{cls[m]['accuracy']:.1f}%", f"{cls[m]['precision']:.1f}%", f"{cls[m]['recall']:.1f}%", f"{cls[m]['f1']:.1f}%", f"{cls[m]['errorRate']:.1f}%"], m == chall) for m in MODELS)
    fam_rows = "".join(tr([k, f"{v.Current:.1f}%", f"{v.LightGBM:.1f}%"]) for k, v in fam.sort_values("LightGBM").iterrows())
    stock_rows = "".join(tr([k, f"{o:.1f}%", f"{s:.1f}%", f"{o + s:.1f}%"]) for k, (o, s) in stock_pol.items())
    pos = int(truth.sum())
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>FreshLoop forecast model</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;color:{INK};background:#f5f6f4;margin:0}} main{{max-width:1080px;margin:0 auto;padding:32px 24px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 10px}} .sub{{color:#5b6762;margin:0 0 18px}}
.card{{background:#fff;border:1px solid #dde3df;border-radius:12px;padding:18px;margin:12px 0}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} .grid3{{display:grid;grid-template-columns:2fr 3fr;gap:12px}} img{{width:100%}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .kpi b{{display:block;font-size:26px}} .kpi span{{color:#5b6762;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border-bottom:1px solid #e6ebe8;padding:7px 8px;text-align:left}} th{{color:#5b6762;font-weight:600}}
.pass,.keep{{display:inline-block;padding:2px 10px;border-radius:99px;color:#fff;font-weight:700;margin-right:8px}} .pass{{background:{GREEN}}} .keep{{background:{AMBER}}} a{{color:{GREEN}}}
</style></head><body><main>
<h1>Forecast model: LightGBM vs current model</h1>
<p class="sub">Real sales: {SOURCE}. Trained on {metrics['trainedOn']['rows']:,} daily records ({metrics['trainedOn']['from']} to {metrics['trainedOn']['to']}); tested on the next {a.test_days} days ({metrics['testedOn']['rows']:,} next-day forecasts) that the models never saw.</p>
<div class="card">{verdict}</div>
<div class="kpis">
<div class="card kpi"><b>{rl['accuracy']:.1f}%</b><span>LightGBM forecast accuracy (current: {rc['accuracy']:.1f}%)</span></div>
<div class="card kpi"><b>{rl['errorRate']:.1f}%</b><span>LightGBM error rate (current: {rc['errorRate']:.1f}%)</span></div>
<div class="card kpi"><b>{cl['f1']:.1f}%</b><span>surplus-flag F1 (current: {cc['f1']:.1f}%)</span></div>
<div class="card kpi"><b>{cl['precision']:.1f}%</b><span>surplus-flag precision (current: {cc['precision']:.1f}%)</span></div>
</div>
<h2>1. Forecasting accuracy</h2>
<div class="grid"><div class="card"><img src="data:image/png;base64,{imgs['acc']}" alt="error rate by model"></div>
<div class="card"><table><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>Error rate</th><th>Accuracy</th><th>Bias</th></tr>{reg_rows}</table>
<p class="sub" style="margin-top:12px">MAE and RMSE are in units per store per day. Error rate (WAPE) = total units missed ÷ total units sold. Accuracy = 100 − error rate. Bias above 0 means over-forecasting.</p></div></div>
<div class="card"><img src="data:image/png;base64,{imgs['series']}" alt="example line"></div>
<div class="card"><table><tr><th>Product family</th><th>Current error rate</th><th>LightGBM error rate</th></tr>{fam_rows}</table></div>
<h2>2. Surplus-risk flags (accuracy, precision, recall, F1, error rate)</h2>
<p class="sub">Each morning a store holds its usual order: the same day last week plus 15%. The system flags a store-family line as <b>at risk</b> when its forecast says more than 10% of that stock will be left unsold, so the surplus can be listed for merchants early. {pos:,} of {len(te):,} test days really had a surplus.</p>
<div class="grid3"><div class="card"><img src="data:image/png;base64,{imgs['cm']}" alt="confusion matrix"></div>
<div class="card"><img src="data:image/png;base64,{imgs['cls']}" alt="classification metrics"></div></div>
<div class="card"><table><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 score</th><th>Error rate</th></tr>{cls_rows}</table>
<p class="sub" style="margin-top:12px">Precision: of the lines flagged, how many really had a surplus. Recall: of the lines that had a surplus, how many were flagged. F1 balances the two. "Same day last week" forecasts the same sales as last week, which is always below its own order plus 15%, so it flags every line: recall is 100% but precision is only the share of days that really had a surplus.</p></div>
<h2>3. Stock each policy would hold</h2>
<div class="grid"><div class="card"><img src="data:image/png;base64,{imgs['stock']}" alt="stock check"></div>
<div class="card"><table><tr><th>Policy</th><th>Over-stocked</th><th>Short</th><th>Total missed</th></tr>{stock_rows}</table>
<p class="sub" style="margin-top:12px">P85 covered {coverage:.0f}% of days (target 85%). Units above what sold risk being wasted; units below are stockouts.</p></div></div>
<h2>4. Inside the model</h2>
<div class="grid"><div class="card"><img src="data:image/png;base64,{imgs['imp']}" alt="feature importance"></div>
<div class="card"><img src="data:image/png;base64,{imgs['buf']}" alt="learned buffers"></div></div>
<p class="sub">LightGBM {lgb.__version__}, quantile loss at 0.50 and 0.85, 600 trees, fixed seed. Files: <a href="metrics.json">metrics.json</a> · <a href="forecasts.csv">forecasts.csv</a> · <a href="lightgbm-p50.txt">lightgbm-p50.txt</a> · <a href="lightgbm-p85.txt">lightgbm-p85.txt</a> · <a href="../data-report/index.html">Data quality report</a></p>
</main></body></html>"""
    (out / "index.html").write_text(html)

    md = [f"### Forecast model on real Favorita sales: {'LightGBM selected' if ship else 'current model kept'}",
          "| Model | MAE | RMSE | Error rate | Accuracy |", "| --- | --- | --- | --- | --- |"]
    md += [f"| {m} | {reg[m]['mae']:.1f} | {reg[m]['rmse']:.1f} | {reg[m]['errorRate']:.1f}% | {reg[m]['accuracy']:.1f}% |" for m in MODELS]
    md += ["", "**Surplus-risk flags**", "| Model | Accuracy | Precision | Recall | F1 | Error rate |", "| --- | --- | --- | --- | --- | --- |"]
    md += [f"| {m} | {cls[m]['accuracy']:.1f}% | {cls[m]['precision']:.1f}% | {cls[m]['recall']:.1f}% | {cls[m]['f1']:.1f}% | {cls[m]['errorRate']:.1f}% |" for m in MODELS]
    md += ["", "| Stock policy | Over-stocked | Short |", "| --- | --- | --- |"] + [f"| {k} | {o:.1f}% | {s:.1f}% |" for k, (o, s) in stock_pol.items()]
    md += ["", "Charts: download the **model-report** artifact, or open Forecast model in the app."]
    text = "\n".join(md)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"selected={selected}\n")


if __name__ == "__main__":
    main()
