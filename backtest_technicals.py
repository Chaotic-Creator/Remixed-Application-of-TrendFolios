"""
TrendFolios + Technicals — configurable voter backtest
======================================================
What changed from backtest.py / backtest_nss.py:

  1. The vote threshold is now a FRACTION of the voters, not a fixed count.
     Old rule: "5 yes votes." If you add voters, the bar silently drops.
     New rule: "more than half of however many voters there are."
     This is what makes barebones vs hybrid an honest comparison.

  2. Voters are switched on and off in CONFIG below, so you can run
     several versions and compare them in one go.

  3. Added a MACD voter computed on SMH's raw price (not the ratio).
     Added an optional RSI voter so you can demonstrate its redundancy.

Requires: pip install pandas numpy
"""

import pandas as pd
import numpy as np

# ============================== CONFIG =================================
SMH_CSV   = "BATS_SMH_1D.csv"
SPY_CSV   = "BATS_SPY_1D.csv"
HORIZONS  = [5, 21, 63]      # week / month / quarter, in trading days
VOL_WINDOW = 252             # lookback for volatility and mean
TRADING_DAYS = 252           # days per year, for annualizing

# Each entry is one version of the algorithm to test.
# The values are which voter families to switch on.
VERSIONS = {
    "A. Barebones (momentum + trend)": ["M", "T"],
    "B. Original 9-vote (incl. broken spread)": ["M", "T", "SP"],
    "C. Hybrid (momentum + trend + MACD)": ["M", "T", "MACD"],
    "D. Hybrid + RSI (shows redundancy)": ["M", "T", "MACD", "RSI"],
}

# ============================ LOAD DATA ================================
def load_csv(path, name):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["time"], unit="s").dt.normalize()
    return df.set_index("date")["close"].rename(name)

px = pd.concat([load_csv(SMH_CSV, "SMH"), load_csv(SPY_CSV, "SPY")],
               axis=1, join="inner").sort_index()

# ====================== INDICATOR HELPERS ==============================
def rsi(series, n=14):
    """Relative Strength Index. 0-100. Above 50 = rising lately."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)

def macd_histogram(series, fast=12, slow=26, signal=9):
    """MACD histogram. Positive = short-term average above long-term."""
    macd_line = (series.ewm(span=fast, adjust=False).mean()
                 - series.ewm(span=slow, adjust=False).mean())
    return macd_line - macd_line.ewm(span=signal, adjust=False).mean()

# ======================= BUILD EVERY VOTER =============================
P  = px["SMH"] / px["SPY"]          # price ratio
R1 = P.pct_change() * 100           # daily relative return, %
CR = 100 * (1 + R1 / 100).cumprod() # compounded relative return index
CR.iloc[0] = 100.0

voters = {}
ready  = []

for v in HORIZONS:
    Rv      = CR.pct_change(v) * 100
    MA      = CR.rolling(v).mean()
    sigma_v = Rv.rolling(VOL_WINDOW).std(ddof=1)
    Rv_mean = Rv.rolling(VOL_WINDOW).mean()
    S       = R1 - Rv_mean + sigma_v        # the broken spread, kept for version B

    voters[f"M_{v}"]  = (CR > MA).astype(int)
    voters[f"T_{v}"]  = (MA > MA.shift(1)).astype(int)
    voters[f"SP_{v}"] = (S > 0).astype(int)

    ready.append(Rv.notna() & MA.notna() & sigma_v.notna() & MA.shift(1).notna())

# Technical voters run on SMH's RAW PRICE, not the ratio.
voters["MACD_price"] = (macd_histogram(px["SMH"]) > 0).astype(int)
voters["RSI_price"]  = (rsi(px["SMH"]) > 50).astype(int)

votes_df = pd.DataFrame(voters)
all_ready = pd.concat(ready, axis=1).all(axis=1)
votes_df = votes_df[all_ready]

smh_ret = px["SMH"].pct_change().reindex(votes_df.index)

# ============================ BACKTEST =================================
def run(families, cash_rate=0.0, cost_bps=0.0):
    """cash_rate: annual % earned when flat. cost_bps: cost per trade, basis points."""
    cols = [c for c in votes_df.columns
            if c.rsplit("_", 1)[0] in families or c.split("_")[0] in families]
    yes = votes_df[cols].sum(axis=1)
    signal = (yes > len(cols) / 2).astype(int)     # <-- the fraction rule
    position = signal.shift(1)

    daily_cash = (1 + cash_rate) ** (1 / TRADING_DAYS) - 1
    ret = position * smh_ret + (1 - position) * daily_cash
    trades = position.diff().abs().fillna(0)
    ret = ret - trades * (cost_bps / 10_000)
    ret = ret.dropna()

    eq = (1 + ret).cumprod()
    years = len(ret) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
    dn = ret[ret < 0].std(ddof=1) * np.sqrt(TRADING_DAYS)
    mdd = (eq / eq.cummax() - 1).min()
    return {
        "Voters": len(cols),
        "CAGR %": round(100 * cagr, 1),
        "Vol %": round(100 * vol, 1),
        "Sharpe": round(cagr / vol, 2),
        "Sortino": round(cagr / dn, 2),
        "MaxDD %": round(100 * mdd, 1),
        "Trades": int(trades.sum()),
        "Invested %": round(100 * position.mean(), 0),
        "Final $": round(10_000 * eq.iloc[-1]),
    }

if __name__ == "__main__":
    print(f"Data: {px.index[0].date()} to {px.index[-1].date()}")
    print(f"Live from: {votes_df.index[0].date()} ({len(votes_df)} days)\n")

    for label, cash, cost in [("=== No cash yield, no costs (matches old baseline) ===", 0.0, 0.0),
                              ("=== 4% cash yield, 5bps per trade (realistic) ===", 0.04, 5.0)]:
        print(label)
        rows = {name: run(fams, cash, cost) for name, fams in VERSIONS.items()}
        bh = (1 + smh_ret.reindex(votes_df.index).shift(0)).dropna()
        eq = bh.cumprod(); yrs = len(bh) / TRADING_DAYS
        cg = eq.iloc[-1] ** (1 / yrs) - 1; vl = bh.std(ddof=1) * np.sqrt(TRADING_DAYS)
        rows["Buy & Hold SMH"] = {
            "Voters": 0, "CAGR %": round(100 * cg, 1), "Vol %": round(100 * vl, 1),
            "Sharpe": round(cg / vl, 2), "Sortino": round(cg / bh[bh < 0].std(ddof=1) / np.sqrt(TRADING_DAYS), 2),
            "MaxDD %": round(100 * (eq / eq.cummax() - 1).min(), 1), "Trades": 0,
            "Invested %": 100, "Final $": round(10_000 * eq.iloc[-1]),
        }
        print(pd.DataFrame(rows).T.to_string(), "\n")
