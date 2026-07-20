"""
Custom TrendFolios Backtest — SMH/SPY (Paper Spread Signal Version)
==========================================================
Pipeline (your 11-step remix, with Dr. Rojas' explicit spread equation):
  1. Price ratio P = SMH / SPY (using adjusted closing prices)
  2. Daily relative return R1 (%)
  3. Compounded relative return index CR (base 100)
  4. Per-horizon normalized returns R^v (v-day % change of CR)
  5. Moving averages of CR per horizon
  6. Relative volatility (corrected eq. 5: squared deviations, sum to n-1)
  7. Paper spread signal (eq. 6): S = (R1 - mean(R^v)) / mean(R^v) + sigma^v / 100
  8. Three Binary voters per horizon: Momentum, Trend, Spread
  9. Majority vote: 3 voters x 3 horizons = 9 votes; >4.5 (i.e. >=5) -> hold
 10. Position(t+1) = Signal(t)   <- trading one-day delay
 11. Long/flat SMH equity curve vs buy-and-hold benchmarks

Requires: pip install pandas numpy matplotlib
"""

# This version differs from the normal backtest.py in a sense that it uses the the paper spread signal equation (eq. 6) 
# instead of the custom additive remix. 



import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save chart data to file png
import matplotlib.pyplot as plt

# ----------------------------- CONFIG ---------------------------------
SMH_CSV = "BATS_SMH_1D.csv"
SPY_CSV = "BATS_SPY_1D.csv"

HORIZONS = [5, 21, 63]     # v: week / month / 3 months
VOL_WINDOW = 252           # n for the volatility & mean stack (trailing obs of R^v)
STARTING_CAPITAL = 10_000  # cosmetic only — scales the curve, not the stats
VOTES_NEEDED = 5           # majority of 9 votes (> 4.5)
TRADING_DAYS = 252         # annualization constant for later testing

# Trading will begin with algorithnm during April of 2021, 63 + 252 = 314 days to finish warm-up.
# 252 for valid Normalized Returns, 63 for longest time horizon (3 months)
# Valid Normalized Returns are required to have reading of Volatility to compute the Spread Signal

# ----------------------------- LOAD DATA ------------------------------
def load_tradingview_csv(path: str, name: str) -> pd.Series:
    """Load a TradingView export; return adjusted close indexed by date."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["time"], unit="s").dt.normalize()
    s = df.set_index("date")["close"].rename(name)
    return s

smh = load_tradingview_csv(SMH_CSV, "SMH")
spy = load_tradingview_csv(SPY_CSV, "SPY")

# Inner join: only dates where BOTH tickers traded (SPY file starts in 2015, SMH in 2020)
# The join trims automatically to the overlap).
px = pd.concat([smh, spy], axis=1, join="inner").sort_index()
print(f"Aligned data: {len(px)} trading days, "
      f"{px.index[0].date()} -> {px.index[-1].date()}")

# ----------------------- SIGNAL PIPELINE (steps 1-7) -------------------
# Step 1: price ratio
P = px["SMH"] / px["SPY"]

# Step 2: daily relative return (%)
R1 = P.pct_change() * 100

# Step 3: compounded relative return index, base 100
CR = 100 * (1 + R1 / 100).cumprod()
CR.iloc[0] = 100.0

sig = pd.DataFrame(index=px.index)

for v in HORIZONS:
    # Step 4: normalized v-day return of the CR index (%)
    Rv = CR.pct_change(v) * 100

    # Step 5: moving average of CR over v days (window ends at t — backward only)
    MA = CR.rolling(v).mean()

    # Step 6: relative volatility w. corrected formula
    # over the trailing VOL_WINDOW observations of R^v
    sigma_v = Rv.rolling(VOL_WINDOW).std(ddof=1)

    # mean of R^v over the same trailing window
    Rv_mean = Rv.rolling(VOL_WINDOW).mean()

    # Step 7: spread signal with Dr. Rojas' explicit equation 6
    # S = (R1 - mean(R^v)) / mean(R^v) + sigma^v / 100
    # WARNING: dividing by mean(R^v) is unstable when the mean is near zero
    # (blow-up) or negative (sign inversion) — this is the instability the
    # custom additive remix was designed to avoid.
    S = (R1 - Rv_mean) / Rv_mean + sigma_v / 100

    # Step 8: the three binary voters (NaN-safe: comparisons on NaN give
    # False, but we trim all warm-up rows below so no fake votes survive)
    sig[f"M_{v}"] = (CR > MA).astype(int)
    sig[f"T_{v}"] = (MA > MA.shift(1)).astype(int)
    sig[f"SP_{v}"] = (S > 0).astype(int)

    # track where the raw inputs are still NaN (warm-up detection)
    sig[f"_ready_{v}"] = (~Rv.isna()) & (~MA.isna()) & (~sigma_v.isna()) & (~MA.shift(1).isna())

# --------------------- WARM-UP TRIM + VOTE (steps 8-9) -----------------
ready_cols = [c for c in sig.columns if c.startswith("_ready_")]
all_ready = sig[ready_cols].all(axis=1)
first_live = all_ready.idxmax()  # first date where every voter is genuine
print(f"All voters live from: {first_live.date()} "
      f"(warm-up = {int((~all_ready).sum())} days)")

vote_cols = [c for c in sig.columns if not c.startswith("_")]
votes = sig.loc[all_ready, vote_cols].sum(axis=1)

# Step 9: majority of 9
signal = (votes >= VOTES_NEEDED).astype(int)

# ------------------------ BACKTEST (steps 10-11) -----------------------
# Step 10: the lag: today's signal sets TOMORROW's position
position = signal.shift(1)

bt = pd.DataFrame(index=signal.index)
bt["smh_ret"] = px["SMH"].pct_change().reindex(bt.index)
bt["spy_ret"] = px["SPY"].pct_change().reindex(bt.index)
bt["position"] = position
bt = bt.dropna()  # drops the first row (no lagged signal yet)

# Long/flat: earn SMH's return when position = 1, 0% cash when 0
bt["strat_ret"] = bt["position"] * bt["smh_ret"]

bt["strategy"] = STARTING_CAPITAL * (1 + bt["strat_ret"]).cumprod()
bt["bh_smh"] = STARTING_CAPITAL * (1 + bt["smh_ret"]).cumprod()
bt["bh_spy"] = STARTING_CAPITAL * (1 + bt["spy_ret"]).cumprod()

# ---------------------------- METRICS ----------------------------------
def metrics(returns: pd.Series, equity: pd.Series) -> dict:
    n = len(returns)
    years = n / TRADING_DAYS
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1
    ann_vol = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    # Paper-style Sharpe/Sortino: no risk-free rate (pure return/risk)
    sharpe = cagr / ann_vol if ann_vol > 0 else np.nan
    downside = returns[returns < 0].std(ddof=1) * np.sqrt(TRADING_DAYS)
    sortino = cagr / downside if downside > 0 else np.nan
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "Final Value ($)": f"{equity.iloc[-1]:,.0f}",
        "Total Return": f"{total:.1%}",
        "CAGR": f"{cagr:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe (no rf)": f"{sharpe:.2f}",
        "Sortino (no rf)": f"{sortino:.2f}",
        "Max Drawdown": f"{max_dd:.1%}",
        "Calmar": f"{calmar:.2f}",
    }

table = pd.DataFrame({
    "Strategy (long/flat)": metrics(bt["strat_ret"], bt["strategy"]),
    "Buy & Hold SMH": metrics(bt["smh_ret"], bt["bh_smh"]),
    "Buy & Hold SPY": metrics(bt["spy_ret"], bt["bh_spy"]),
})

trades = int((bt["position"].diff().abs() == 1).sum())
days_invested = bt["position"].mean()

print("\n================ PERFORMANCE ================")
print(table.to_string())
print("==============================================")
print(f"Round-trip flips (buys+sells): {trades}")
print(f"Time invested: {days_invested:.1%} of days")
print(f"Backtest span: {bt.index[0].date()} -> {bt.index[-1].date()} "
      f"({len(bt)} days)")

# ----------------------------- CHART -----------------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(11, 7), sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)
ax1.plot(bt.index, bt["strategy"], label="Strategy (long/flat SMH)", lw=1.6)
ax1.plot(bt.index, bt["bh_smh"], label="Buy & Hold SMH", lw=1.2, alpha=0.8)
ax1.plot(bt.index, bt["bh_spy"], label="Buy & Hold SPY", lw=1.2, alpha=0.8)
ax1.set_ylabel(f"Growth of ${STARTING_CAPITAL:,}")
ax1.set_title("Custom TrendFolios: 3-vote algorithm, horizons 5/21/63 (Paper Spread Eq.)")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.fill_between(bt.index, bt["position"], step="post", alpha=0.4)
ax2.set_ylabel("Position")
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["Cash", "SMH"])
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("backtest_result_paper.png", dpi=150)
print("\nChart saved to backtest_result_paper.png")