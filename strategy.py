"""Global Trend Guard Portfolio: data, backtest engine and performance measures.

The functions here are the same as in our notebook (Global_Trend_Guard_Backtest.ipynb),
so the app reproduces the prospectus numbers with the default settings.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

RISKY = ['SPY', 'EFA', 'EEM', 'IEF', 'GLD']
CASH = 'SHY'
SPX = '^SP500TR'
RF_TICKER = '^IRX'
EQUITY = ['SPY', 'EFA', 'EEM']

BASE_WEIGHT = 1 / len(RISKY)
DEFAULT_WINDOW = 10
DEFAULT_TC_BP = 10

DATA_START = '2004-11-01'
DATA_END = '2026-01-01'          # exclusive -> data up to 31 Dec 2025
BT_END = '2025-12-31'


# ---------------------------------------------------------------- data
def download_monthly(retries=3):
    """Daily adjusted closes from Yahoo Finance, sampled at each month-end."""
    tickers = RISKY + [CASH, SPX, RF_TICKER]
    missing = tickers
    for _ in range(retries):
        data = yf.download(tickers, start=DATA_START, end=DATA_END, interval='1d',
                           auto_adjust=True, progress=False)
        close = data['Close']
        missing = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
        if not missing:
            if close.index.tz is not None:
                close.index = close.index.tz_localize(None)
            return close[tickers].resample('ME').last()
    raise RuntimeError(f'Yahoo Finance returned no data for {missing}.')


def prepare(monthly):
    prices = monthly[RISKY + [CASH]]
    rets = prices.pct_change()
    spx_rets = monthly[SPX].pct_change()
    rf = (monthly[RF_TICKER] / 100 / 12).shift(1)   # yield known at end of previous month
    return prices, rets, spx_rets, rf


# ---------------------------------------------------------------- strategy
def target_weights(prices, window=DEFAULT_WINDOW, base_weight=BASE_WEIGHT):
    """Weights decided at each month-end, using only prices up to that month-end."""
    px = prices[RISKY]
    sma = px.rolling(window).mean()
    signal = (px > sma).astype(float).where(sma.notna())
    w = signal * base_weight
    w[CASH] = 1 - w.sum(axis=1, min_count=len(RISKY))
    return w, signal, sma


def static_weights(index):
    w = pd.DataFrame(BASE_WEIGHT, index=index, columns=RISKY)
    w[CASH] = 0.0
    return w


def run_backtest(w_target, rets, tc):
    """Gross/net monthly returns and turnover; decisions at t earn the return of t+1."""
    r = rets.loc[w_target.index, w_target.columns]
    w_held = w_target.shift(1)
    gross = (w_held * r).sum(axis=1, min_count=1)
    w_drift = w_held * (1 + r)
    w_drift = w_drift.div(w_drift.sum(axis=1), axis=0)
    turnover = (w_target - w_drift).abs().sum(axis=1)
    cost = tc * turnover
    out = pd.DataFrame({'gross': gross, 'turnover': turnover, 'cost': cost})
    out['net'] = out['gross'] - out['cost']
    return out.iloc[1:]


def backtest(monthly, window=DEFAULT_WINDOW, tc_bp=DEFAULT_TC_BP, bt_start='2006-01-31', bt_end=BT_END):
    prices, rets, spx_rets, rf = prepare(monthly)
    dec_start = (pd.Timestamp(bt_start) - pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')
    w, signal, sma = target_weights(prices, window)
    w = w.loc[dec_start:bt_end]
    if w.isna().any().any():
        raise ValueError('Not enough price history for this window and start date.')
    tc = tc_bp / 10000
    bt_s = run_backtest(w, rets, tc)
    bt_e = run_backtest(static_weights(w.index), rets, tc)
    R = pd.DataFrame({'Strategy': bt_s['net'], 'Strategy (gross)': bt_s['gross'],
                      'Static 20%': bt_e['net'], 'S&P 500 TR': spx_rets}).loc[bt_start:bt_end]
    return {'R': R, 'rf': rf.loc[R.index], 'weights': w, 'held': w.shift(1).loc[bt_start:bt_end],
            'signal': signal, 'sma': sma, 'prices': prices, 'turnover_s': bt_s['turnover'],
            'turnover_e': bt_e['turnover'], 'start_date': w.index[0]}


# ---------------------------------------------------------------- measures
def max_drawdown(r):
    v = (1 + r).cumprod()
    hwm = v.cummax().clip(lower=1.0)
    return (1 - v / hwm).max()


def regression_stats(r, bench, rf):
    ex, exb = r - rf, bench - rf
    model = sm.OLS(ex, sm.add_constant(exb)).fit()
    a, b = model.params.iloc[0], model.params.iloc[1]
    resid_sd = np.sqrt(model.mse_resid)
    return {'alpha': 12 * a, 'alpha_t': model.tvalues.iloc[0], 'beta': b,
            'IR': (12 * a) / (np.sqrt(12) * resid_sd)}


def basic_stats(r, rf):
    ex = r - rf
    return {'Average return p.a.': 12 * r.mean(),
            'Volatility p.a.': np.sqrt(12) * r.std(),
            'Sharpe ratio': 12 * ex.mean() / (np.sqrt(12) * ex.std()),
            'CAGR': (1 + r).prod() ** (12 / len(r)) - 1,
            'Maximum drawdown': max_drawdown(r)}


def perf_table(res):
    R, rf = res['R'], res['rf']
    cols = ['Strategy', 'Static 20%', 'S&P 500 TR']
    rows = {c: basic_stats(R[c], rf) for c in cols}
    for label, bcol in [('static 20%', 'Static 20%'), ('S&P 500', 'S&P 500 TR')]:
        for c in cols:
            if c == bcol:
                continue
            s = regression_stats(R[c], R[bcol], rf)
            rows[c][f'Alpha p.a. vs {label}'] = s['alpha']
            rows[c][f't-stat of alpha vs {label}'] = s['alpha_t']
            rows[c][f'Beta vs {label}'] = s['beta']
            rows[c][f'Information ratio vs {label}'] = s['IR']
    t = pd.DataFrame(rows)
    t.loc['Average annual turnover', 'Strategy'] = 12 * res['turnover_s'].mean()
    t.loc['Average annual turnover', 'Static 20%'] = 12 * res['turnover_e'].mean()
    return t


def wealth(R, start_date):
    start = pd.DataFrame(0.0, index=[start_date], columns=R.columns)
    return (1 + pd.concat([start, R])).cumprod()
