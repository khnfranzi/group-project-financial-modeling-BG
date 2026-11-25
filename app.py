"""Global Trend Guard Portfolio: interactive backtest (2270 Financial Modeling group assignment)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import strategy as stg

st.set_page_config(page_title='Global Trend Guard Portfolio', layout='wide')

COLORS = {'Strategy': '#257de0', 'Static 20%': '#f47230', 'S&P 500 TR': '#00b894'}
ASSET_COLORS = {'SPY': '#257de0', 'EFA': '#00b894', 'EEM': '#6a4ce2',
                'IEF': '#f2a600', 'GLD': '#ee7fac', 'SHY': '#c8c8c8'}
SERIES = ['Strategy', 'Static 20%', 'S&P 500 TR']


@st.cache_data(ttl=24 * 3600, show_spinner='Downloading prices from Yahoo Finance ...')
def load_data():
    return stg.download_monthly()


def layout(fig, title, yfmt=None, height=420):
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)), height=height,
                      margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor='white',
                      legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1),
                      hovermode='x unified')
    fig.update_xaxes(showgrid=False, linecolor='#999')
    fig.update_yaxes(gridcolor='#ececec', zeroline=False, tickformat=yfmt)
    return fig


def fmt_table(t):
    out = t.copy().astype(object)
    for r in t.index:
        for c in t.columns:
            v = t.loc[r, c]
            if pd.isna(v):
                out.loc[r, c] = 'n/a'
            elif any(k in r for k in ['ratio', 'Beta', 't-stat']):
                out.loc[r, c] = f'{v:.2f}'
            else:
                out.loc[r, c] = f'{v:.1%}'
    return out


# ---------------------------------------------------------------- sidebar
st.sidebar.header('Assumptions')
window = st.sidebar.slider('Moving-average window (months)', 3, 12, stg.DEFAULT_WINDOW,
                           help='An ETF is held only if its month-end price is above its average over this many months.')
tc_bp = st.sidebar.slider('Transaction cost (basis points per unit of turnover)', 0, 50, stg.DEFAULT_TC_BP, step=5,
                          help='Applied to the strategy and to the static benchmark at every monthly rebalance.')
start_year = st.sidebar.selectbox('Backtest starts in January of', list(range(2006, 2021)), index=0)
is_default = (window == stg.DEFAULT_WINDOW and tc_bp == stg.DEFAULT_TC_BP and start_year == 2006)
st.sidebar.caption('Default settings (10 months, 10 bp, 2006) reproduce the numbers in our prospectus.')
st.sidebar.divider()
st.sidebar.caption('Data: Yahoo Finance via yfinance, monthly, January 2006 to December 2025. '
                   'Results are backtested and hypothetical and are not investment advice.')

# ---------------------------------------------------------------- data + backtest
try:
    monthly = load_data()
except Exception as e:  # noqa: BLE001
    st.error(f'Could not download data from Yahoo Finance ({e}). Please reload the page in a minute.')
    st.stop()

res = stg.backtest(monthly, window=window, tc_bp=tc_bp, bt_start=f'{start_year}-01-31')
R, rf = res['R'], res['rf']
table = stg.perf_table(res)

# ---------------------------------------------------------------- header
st.title('Global Trend Guard Portfolio')
st.markdown('Five global ETFs at 20% each. At every month-end, an ETF stays in the portfolio only if its price is above '
            f'its **{window}-month** moving average; otherwise its 20% moves into short-term Treasuries (SHY). '
            'Change the assumptions on the left to see how performance and the benchmark comparison respond.')
if not is_default:
    st.info('You are looking at modified assumptions. The prospectus uses 10 months, 10 bp and a 2006 start.')

c = st.columns(5)
s, e = table['Strategy'], table['Static 20%']
kpis = [('Average return p.a.', f"{s['Average return p.a.']:.1%}", f"Static 20%: {e['Average return p.a.']:.1%}"),
        ('Volatility p.a.', f"{s['Volatility p.a.']:.1%}", f"Static 20%: {e['Volatility p.a.']:.1%}"),
        ('Sharpe ratio', f"{s['Sharpe ratio']:.2f}", f"Static 20%: {e['Sharpe ratio']:.2f}"),
        ('Maximum drawdown', f"{s['Maximum drawdown']:.1%}", f"Static 20%: {e['Maximum drawdown']:.1%}"),
        ('Alpha p.a. vs static', f"{s['Alpha p.a. vs static 20%']:.1%}", f"Beta vs static: {s['Beta vs static 20%']:.2f}")]
for col, (label, value, note) in zip(c, kpis):
    col.metric(label, value)
    col.caption(note)

tab1, tab2, tab3, tab4 = st.tabs(['Performance', 'Allocation', 'Risk', 'How it works'])

# ---------------------------------------------------------------- performance
with tab1:
    W = stg.wealth(R[SERIES], res['start_date'])
    log = st.toggle('Logarithmic scale', value=False)
    fig = go.Figure()
    for name in SERIES:
        fig.add_trace(go.Scatter(x=W.index, y=W[name], name=name, line=dict(color=COLORS[name], width=2.4 if name == 'Strategy' else 1.8),
                                 hovertemplate='%{y:.2f}'))
    layout(fig, f'Growth of $1 invested at the end of {res["start_date"]:%B %Y} (after costs)', yfmt='.2f')
    if log:
        fig.update_yaxes(type='log')
    st.plotly_chart(fig, width='stretch')

    left, right = st.columns([3, 2])
    with left:
        st.subheader('Performance measures')
        st.dataframe(fmt_table(table), width='stretch', height=560)
    with right:
        st.subheader('How to read the table')
        st.markdown(
            '- **Average return and volatility** are annualised from monthly returns (×12 and ×√12).\n'
            '- **Sharpe ratio** uses excess returns over the 13-week T-bill.\n'
            '- **Alpha and beta** come from regressing the strategy\'s excess returns on the benchmark\'s. '
            'Alpha is annualised (×12); beta is not.\n'
            '- **Information ratio** is alpha divided by the volatility of the regression residuals, annualised.\n'
            '- The strategy and the static portfolio are shown after transaction costs; the S&P 500 index has no costs.')
        gross = R['Strategy (gross)']
        st.caption(f"Before costs the strategy returned {12 * gross.mean():.1%} a year. Costs reduced this by "
                   f"{12 * (gross - R['Strategy']).mean():.2%} a year at {tc_bp} bp.")

# ---------------------------------------------------------------- allocation
with tab2:
    held = res['held']
    fig = go.Figure()
    for a in stg.RISKY + [stg.CASH]:
        fig.add_trace(go.Scatter(x=held.index, y=held[a], name=a if a != 'SHY' else 'SHY (cash)', stackgroup='w',
                                 mode='lines', line=dict(width=0, shape='hv', color=ASSET_COLORS[a]),
                                 fillcolor=ASSET_COLORS[a], hovertemplate='%{y:.0%}'))
    layout(fig, 'Weights held each month', yfmt='.0%')
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, width='stretch')

    a1, a2 = st.columns(2)
    with a1:
        st.subheader('Share of months each ETF was held')
        share = held[stg.RISKY].gt(0).mean().rename('Months held').to_frame()
        share.loc['SHY (avg. weight)'] = held[stg.CASH].mean()
        st.dataframe(share.style.format('{:.0%}'), width='stretch')
        st.caption(f"Average equity weight: {held[stg.EQUITY].sum(axis=1).mean():.1%} (static portfolio: 60%).")
    with a2:
        last = res['weights'].index[-1]
        st.subheader(f'Signal at {last:%d %B %Y}')
        sig = pd.DataFrame({'Price': res['prices'].loc[last, stg.RISKY],
                            f'{window}-month average': res['sma'].loc[last, stg.RISKY]})
        sig['Above trend'] = np.where(sig['Price'] > sig[f'{window}-month average'], 'yes, hold', 'no, in SHY')
        sig['Weight next month'] = res['weights'].loc[last, stg.RISKY].map('{:.0%}'.format)
        st.dataframe(sig.style.format({'Price': '{:.2f}', f'{window}-month average': '{:.2f}'}), width='stretch')

# ---------------------------------------------------------------- risk
with tab3:
    W = stg.wealth(R[SERIES], res['start_date'])
    dd = W / W.cummax() - 1
    fig = go.Figure()
    for name in SERIES:
        fig.add_trace(go.Scatter(x=dd.index, y=dd[name], name=name, line=dict(color=COLORS[name], width=1.8),
                                 hovertemplate='%{y:.1%}'))
    layout(fig, 'Drawdown from the previous peak', yfmt='.0%', height=380)
    st.plotly_chart(fig, width='stretch')

    r1, r2 = st.columns([2, 3])
    with r1:
        st.subheader('Stress periods')
        periods = {'Financial crisis (Nov 2007 to Feb 2009)': ('2007-11-30', '2009-02-28'),
                   'COVID crash (Feb to Mar 2020)': ('2020-02-29', '2020-03-31'),
                   'Rate shock (Jan to Dec 2022)': ('2022-01-31', '2022-12-31')}
        rows = {k: (1 + R.loc[a:b, SERIES]).prod() - 1 for k, (a, b) in periods.items()
                if pd.Timestamp(a) >= R.index[0]}
        if rows:
            st.dataframe(pd.DataFrame(rows).T.style.format('{:.1%}'), width='stretch')
        else:
            st.caption('No stress period falls inside the selected sample.')
    with r2:
        yearly = (1 + R[SERIES]).groupby(R.index.year).prod() - 1
        fig = go.Figure()
        for name in SERIES:
            fig.add_trace(go.Bar(x=yearly.index, y=yearly[name], name=name, marker_color=COLORS[name],
                                 hovertemplate='%{y:.1%}'))
        layout(fig, 'Calendar-year returns', yfmt='.0%', height=360)
        fig.update_layout(barmode='group', bargap=0.25)
        st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------- explanation
with tab4:
    st.markdown(f"""
**Who it is for.** Investors roughly 5 to 15 years from retirement who still want growth but cannot afford a deep loss
shortly before they start drawing on their savings (sequence-of-returns risk).

**Universe.** SPY (US equities), EFA (developed markets outside North America), EEM (emerging markets),
IEF (7 to 10 year US Treasuries) and GLD (gold), with SHY (1 to 3 year Treasuries) as the cash position.

**Rule.** On the last trading day of each month, each ETF gets 20% if its dividend-adjusted price is above its
{window}-month simple moving average and 0% otherwise. Whatever is not invested goes into SHY. The weights are set
with month-end prices and earn the following month's return, so no decision uses information that was not yet
available. The portfolio is brought back to its target weights every month.

**Benchmarks.** The same five ETFs at fixed 20% weights, rebalanced monthly with the same costs (primary), and the
S&P 500 total return index (secondary).

**Data and method.** Daily adjusted closes from Yahoo Finance, sampled at month-end; simple returns; transaction costs
equal to the cost rate times turnover; risk-free rate from the 13-week T-bill yield. The calculations are identical to
our Colab notebook, and the default settings reproduce the prospectus.
""")
    st.caption('2270 Financial Modeling, Nova School of Business and Economics. Academic project; backtested, '
               'hypothetical results; not investment advice.')
