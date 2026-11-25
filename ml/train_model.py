"""Trains the LightGBM demand forecast and decides whether it replaces the current model.

Champion:   exponential smoothing + fixed 15% safety buffer (the model in stores today).
Challenger: LightGBM with quantile loss. P50 is the most likely sale; P85 is the stock level we
            are 85% sure covers demand, so every product gets its own safety margin.

Both are trained on 84 days of history and tested on the latest week, which neither has seen
(next-day forecasts). The challenger ships only if it is more accurate AND leaves fewer units
over-stocked or short. Otherwise the champion stays.

  python ml/train_model.py --out artifacts/model-report
"""
import argparse
import base64
import io
import json
import os
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GREEN, CLAY, AMBER, GREY, INK = "#1f5a46", "#c4553a", "#c9922e", "#9aa5a0", "#1d2622"
FIXED_BUFFER = 0.15  # the current policy (server/config/risk-policy.json)
SEED = 42
FEATURES = ["lag_1", "lag_2", "lag_7", "lag_14", "mean_7", "mean_28", "std_7", "dow", "weekend", "price", "site", "sku", "category"]
FEATURE_NAMES = {
    "lag_1": "Sales yesterday", "lag_2": "Sales 2 days ago", "lag_7": "Same day last week", "lag_14": "Same day 2 weeks ago",
    "mean_7": "7-day average", "mean_28": "28-day average", "std_7": "7-day variability", "dow": "Day of week",
    "weekend": "Weekend", "price": "Price", "site": "Store", "sku": "Product", "category": "Category",
}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--history", default=ROOT / "server/data/sales-history.csv")
    p.add_argument("--new", default=ROOT / "server/data/new-sales.csv")
    p.add_argument("--out", default=ROOT / "artifacts/model-report")
    return p.parse_args()


def build_features(hist, new):
    df = pd.concat([hist.assign(split="train"), new.assign(split="test")], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["site", "sku", "date"]).reset_index(drop=True)
    g = df.groupby(["site", "sku"])["units_sold"]
    for k in [1, 2, 7, 14]:
        df[f"lag_{k}"] = g.shift(k)
    past = g.shift(1)  # only days before the one being forecast
    df["mean_7"] = past.groupby([df.site, df.sku]).transform(lambda s: s.rolling(7).mean())
    df["mean_28"] = past.groupby([df.site, df.sku]).transform(lambda s: s.rolling(28).mean())
    df["std_7"] = past.groupby([df.site, df.sku]).transform(lambda s: s.rolling(7).std())
    df["exp_smooth"] = past.groupby([df.site, df.sku]).transform(lambda s: s.ewm(alpha=0.5, adjust=False).mean())
    df["dow"] = df["date"].dt.dayofweek
    df["weekend"] = (df["dow"] >= 5).astype(int)
    for c in ["site", "sku", "category"]:
        df[c] = df[c].astype("category")
    return df.dropna(subset=["mean_28", "lag_14"]).reset_index(drop=True)


def train(train_df, alpha):
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=alpha, n_estimators=400, learning_rate=0.04, num_leaves=15,
        min_child_samples=10, subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
        random_state=SEED, deterministic=True, force_col_wise=True, verbose=-1,
    )
    model.fit(train_df[FEATURES], train_df["units_sold"], categorical_feature=["site", "sku", "category"])
    return model


def wape(actual, forecast):
    return float(100 * np.abs(actual - forecast).sum() / actual.sum())


def stock_check(actual, keep):
    """Units left over (risk of waste) and units short (stockouts), as % of demand."""
    over = np.maximum(0, keep - actual).sum()
    short = np.maximum(0, actual - keep).sum()
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


def label_bars(ax, bars, fmt="{:.1f}%"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height() if hasattr(b, "get_height") else 0), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8, color=INK, xytext=(0, 2), textcoords="offset points")


def main():
    a = args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hist, new = pd.read_csv(a.history), pd.read_csv(a.new)
    df = build_features(hist, new)
    tr, te = df[df.split == "train"].copy(), df[df.split == "test"].copy()

    p50, p85 = train(tr, 0.5), train(tr, 0.85)
    te["lgbm_p50"] = p50.predict(te[FEATURES]).clip(min=0)
    te["lgbm_p85"] = np.maximum(p85.predict(te[FEATURES]), te["lgbm_p50"])
    y = te["units_sold"].to_numpy(float)

    accuracy = {
        "7-day average": wape(y, te["mean_7"]),
        "Same day last week": wape(y, te["lag_7"]),
        "Exponential smoothing (current)": wape(y, te["exp_smooth"]),
        "LightGBM (P50)": wape(y, te["lgbm_p50"]),
    }
    policies = {
        "Current: exp. smoothing + 15% buffer": np.ceil(te["exp_smooth"] * (1 + FIXED_BUFFER)),
        "LightGBM P50 + 15% buffer": np.ceil(te["lgbm_p50"] * (1 + FIXED_BUFFER)),
        "LightGBM P85 (learned buffer)": np.ceil(te["lgbm_p85"]),
    }
    stock = {k: stock_check(y, v.to_numpy(float)) for k, v in policies.items()}
    coverage = float(100 * (y <= np.ceil(te["lgbm_p85"])).mean())

    champ_acc, chall_acc = accuracy["Exponential smoothing (current)"], accuracy["LightGBM (P50)"]
    champ_miss = sum(stock["Current: exp. smoothing + 15% buffer"])
    chall_miss = sum(stock["LightGBM P85 (learned buffer)"])
    selected = "lightgbm-quantile" if (chall_acc < champ_acc and chall_miss < champ_miss) else "exp-smoothing"

    # Learned buffer per product: how much above the likely sale (P50) the model keeps (P85).
    te["buffer_pct"] = 100 * (te["lgbm_p85"] / te["lgbm_p50"].clip(lower=0.1) - 1)
    buffers = te.groupby("sku", observed=True)["buffer_pct"].mean().sort_values()

    p50.booster_.save_model(str(out / "lightgbm-p50.txt"))
    p85.booster_.save_model(str(out / "lightgbm-p85.txt"))
    te[["date", "site", "sku", "units_sold", "exp_smooth", "lgbm_p50", "lgbm_p85"]].assign(date=te.date.dt.date).to_csv(out / "forecasts.csv", index=False)
    metrics = {
        "trainedOn": {"rows": int(len(tr)), "from": str(tr.date.min().date()), "to": str(tr.date.max().date())},
        "testedOn": {"rows": int(len(te)), "from": str(te.date.min().date()), "to": str(te.date.max().date())},
        "wapePct": {k: round(v, 2) for k, v in accuracy.items()},
        "stockCheckPct": {k: {"overStocked": round(o, 2), "short": round(s, 2)} for k, (o, s) in stock.items()},
        "p85CoveragePct": round(coverage, 1),
        "learnedBufferPct": {k: round(v, 1) for k, v in buffers.items()},
        "selected": selected,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---------- charts
    imgs = {}
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    names = list(accuracy)
    bars = ax.bar(range(len(names)), [accuracy[n] for n in names], color=[GREY, GREY, AMBER, GREEN], width=0.6)
    label_bars(ax, bars)
    ax.set_xticks(range(len(names)), [n.replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylabel("Forecast error (WAPE %)")
    ax.set_title("Forecast error on the unseen week (lower is better)", fontsize=10, loc="left")
    style(ax)
    imgs["acc"] = png(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    pn = list(stock)
    x = np.arange(len(pn))
    b1 = ax.bar(x - 0.18, [stock[p][0] for p in pn], 0.36, color=AMBER, label="Over-stocked (risk of waste)")
    b2 = ax.bar(x + 0.18, [stock[p][1] for p in pn], 0.36, color=CLAY, label="Short (stockouts)")
    label_bars(ax, list(b1) + list(b2))
    ax.set_xticks(x, [p.replace(": ", ":\n").replace(" + ", "\n+ ").replace(" (", "\n(") for p in pn], fontsize=8)
    ax.set_ylabel("% of demand")
    ax.set_title("Stock level each policy would hold", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    imgs["stock"] = png(fig)

    key = ("koramangala", "MILK-TND-500")
    s_all = df[(df.site == key[0]) & (df.sku == key[1])].tail(35)
    s_te = s_all[s_all.split == "test"].merge(te[["date", "site", "sku", "lgbm_p50", "lgbm_p85"]], on=["date", "site", "sku"])
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(s_all.date, s_all.units_sold, color=GREY, lw=1.4, marker="o", ms=2.5, label="Actual sales")
    ax.fill_between(s_te.date, s_te.lgbm_p50, s_te.lgbm_p85, color=GREEN, alpha=0.15, label="LightGBM P50 to P85")
    ax.plot(s_te.date, s_te.lgbm_p50, color=GREEN, lw=2, label="LightGBM forecast (P50)")
    ax.plot(s_te.date, s_te.exp_smooth, color=AMBER, lw=1.6, ls="--", label="Exponential smoothing")
    ax.axvline(mdates.date2num(pd.Timestamp(s_te.date.min()).to_pydatetime()) - 0.5, color=GREY, ls=":", lw=1)
    ax.text(s_te.date.min(), ax.get_ylim()[1], " unseen week", va="top", fontsize=8, color="#4a5550")
    ax.set_title("Koramangala · Milk 500 ml: forecasts against what actually sold", fontsize=10, loc="left")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower left")
    style(ax)
    imgs["series"] = png(fig)

    imp = pd.Series(p50.booster_.feature_importance("gain"), index=FEATURES)
    imp = (100 * imp / imp.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.barh([FEATURE_NAMES[f] for f in imp.index], imp.values, color=GREEN)
    ax.set_xlabel("Share of the model's decisions (%)")
    ax.set_title("What the model relies on", fontsize=10, loc="left")
    style(ax, grid="x")
    imgs["imp"] = png(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.barh(buffers.index, buffers.values, color=GREEN)
    ax.axvline(100 * FIXED_BUFFER, color=CLAY, ls="--", lw=1.2, label="Fixed 15% buffer today")
    ax.set_xlabel("Safety margin above the likely sale (%)")
    ax.set_title("Learned safety margin per product (P85 vs P50)", fontsize=10, loc="left")
    ax.set_xlim(0, max(100 * FIXED_BUFFER, buffers.max()) * 1.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax, grid="x")
    imgs["buf"] = png(fig)

    # ---------- report
    ship = selected == "lightgbm-quantile"
    verdict = (f"<span class='pass'>LIGHTGBM SELECTED</span> More accurate ({chall_acc:.1f}% vs {champ_acc:.1f}% error) and fewer units over-stocked or short ({chall_miss:.1f}% vs {champ_miss:.1f}%)."
               if ship else f"<span class='keep'>CURRENT MODEL KEPT</span> LightGBM did not beat exponential smoothing on both accuracy and stock, so it does not ship.")
    rows = "".join(f"<tr><td>{k}</td><td>{v:.1f}%</td></tr>" for k, v in accuracy.items())
    srows = "".join(f"<tr><td>{k}</td><td>{o:.1f}%</td><td>{s:.1f}%</td><td>{o + s:.1f}%</td></tr>" for k, (o, s) in stock.items())
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>FreshLoop forecast model</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;color:{INK};background:#f5f6f4;margin:0}} main{{max-width:1080px;margin:0 auto;padding:32px 24px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 10px}} .sub{{color:#5b6762;margin:0 0 18px}}
.card{{background:#fff;border:1px solid #dde3df;border-radius:12px;padding:18px;margin:12px 0}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} img{{width:100%}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .kpi b{{display:block;font-size:26px}} .kpi span{{color:#5b6762;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border-bottom:1px solid #e6ebe8;padding:7px 8px;text-align:left}} th{{color:#5b6762;font-weight:600}}
.pass,.keep{{display:inline-block;padding:2px 10px;border-radius:99px;color:#fff;font-weight:700;margin-right:8px}} .pass{{background:{GREEN}}} .keep{{background:{AMBER}}} a{{color:{GREEN}}}
</style></head><body><main>
<h1>Forecast model: LightGBM vs current model</h1>
<p class="sub">Trained on {metrics['trainedOn']['rows']} daily sales records across {tr.groupby(['site','sku'], observed=True).ngroups} store-product lines ({metrics['trainedOn']['from']} to {metrics['trainedOn']['to']}); tested on the following week ({metrics['testedOn']['rows']} next-day forecasts) that neither model saw.</p>
<div class="card">{verdict}</div>
<div class="kpis">
<div class="card kpi"><b>{chall_acc:.1f}%</b><span>LightGBM forecast error</span></div>
<div class="card kpi"><b>{champ_acc:.1f}%</b><span>current model's error</span></div>
<div class="card kpi"><b>{coverage:.0f}%</b><span>of days covered by P85 (target 85%)</span></div>
<div class="card kpi"><b>{buffers.min():.0f}–{buffers.max():.0f}%</b><span>learned safety margin, vs fixed 15%</span></div>
</div>
<h2>Accuracy</h2>
<div class="grid"><div class="card"><img src="data:image/png;base64,{imgs['acc']}" alt="error by model"></div>
<div class="card"><table><tr><th>Model</th><th>Error (WAPE)</th></tr>{rows}</table>
<p class="sub" style="margin-top:12px">WAPE = total units the forecast missed by, divided by total units sold.</p></div></div>
<div class="card"><img src="data:image/png;base64,{imgs['series']}" alt="example product"></div>
<h2>Stock: what each policy would hold</h2>
<div class="grid"><div class="card"><img src="data:image/png;base64,{imgs['stock']}" alt="stock check"></div>
<div class="card"><table><tr><th>Policy</th><th>Over-stocked</th><th>Short</th><th>Total missed</th></tr>{srows}</table>
<p class="sub" style="margin-top:12px">Each day, each policy holds a stock level. Units above what sold risk being wasted; units below are stockouts.</p></div></div>
<h2>Inside the model</h2>
<div class="grid"><div class="card"><img src="data:image/png;base64,{imgs['imp']}" alt="feature importance"></div>
<div class="card"><img src="data:image/png;base64,{imgs['buf']}" alt="learned buffers"></div></div>
<p class="sub">LightGBM {lgb.__version__}, quantile loss at 0.50 and 0.85, 400 trees, fixed seed. Files: <a href="metrics.json">metrics.json</a> · <a href="forecasts.csv">forecasts.csv</a> · <a href="lightgbm-p50.txt">lightgbm-p50.txt</a> · <a href="lightgbm-p85.txt">lightgbm-p85.txt</a> · <a href="../data-report/index.html">Data quality report</a></p>
</main></body></html>"""
    (out / "index.html").write_text(html)

    md = [f"### Forecast model: {'LightGBM selected' if ship else 'current model kept'}",
          "| Model | Forecast error (WAPE) |", "| --- | --- |"] + [f"| {k} | {v:.1f}% |" for k, v in accuracy.items()] + [
          "", "| Stock policy | Over-stocked | Short |", "| --- | --- | --- |"] + [f"| {k} | {o:.1f}% | {s:.1f}% |" for k, (o, s) in stock.items()] + [
          "", f"P85 covered {coverage:.0f}% of days (target 85%). Learned safety margin: {buffers.min():.0f}–{buffers.max():.0f}% by product, against a fixed 15%.",
          "", "Charts: download the **model-report** artifact, or open Forecast model in the app."]
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
