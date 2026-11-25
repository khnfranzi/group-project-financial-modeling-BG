# Global Trend Guard Portfolio – interactive backtest

Streamlit app for our 2270 Financial Modeling group assignment (Nova SBE).

**Live app:** [add Streamlit URL]  
**Group:** [group name], [member names]

## What the app does

The portfolio holds five ETFs (SPY, EFA, EEM, IEF, GLD) at 20% each. At every month-end, an ETF is kept only if its
dividend-adjusted price is above its moving average; otherwise its 20% moves into short-term Treasuries (SHY).
The app compares the strategy with the same ETFs at fixed 20% weights and with the S&P 500 total return index.

Users can change three assumptions and see all results update:

- the moving-average window (3 to 12 months),
- the transaction cost per unit of turnover (0 to 50 basis points),
- the start year of the backtest (2006 to 2020).

The app shows cumulative performance, the required performance measures (average return, volatility, Sharpe ratio,
alpha, beta, information ratio, plus CAGR and maximum drawdown), the allocation over time, the current signals,
drawdowns, stress periods and calendar-year returns.

## Consistency with the notebook

`strategy.py` contains the same data handling, backtest engine and performance measures as our Colab notebook
(`Global_Trend_Guard_Backtest.ipynb`). With the default settings (10 months, 10 bp, start in 2006) the app reproduces
the numbers in the prospectus.

## Run locally

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
streamlit run app.py
```

The app downloads data from Yahoo Finance through `yfinance` on first load (cached for 24 hours), so it needs an
internet connection. No data files are stored in the repository.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface |
| `strategy.py` | Data download, signal, backtest engine and performance measures |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Colour theme |

## Disclaimer

Academic project. All results are backtested and hypothetical and are not investment advice.
The app was built with the help of AI coding assistance (Claude, Anthropic); the group reviewed and tested the code.
