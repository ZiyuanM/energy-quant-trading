# %%
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def backtest_pairs(ticker1, ticker2, start_date, end_date):
    # 1. Download Data
    data = yf.download([ticker1, ticker2], start=start_date, end=end_date)
    
    # Extract 'Adj Close' safely
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        # Fallback for some versions where it is 'Close' + auto-adjusted
        prices = data['Close']
        
    S1 = prices[ticker1]
    S2 = prices[ticker2]

    # 2. Test for Cointegration
    score, pvalue, _ = coint(S1, S2)

    # 3. Calculate the Spread and Z-Score
    S1_with_const = sm.add_constant(S1)
    results = sm.OLS(S2, S1_with_const).fit()
    beta = results.params[ticker1]
    spread = S2 - beta * S1
    zscore = (spread - spread.mean()) / np.std(spread)

    # 4. Trading Logic
    df = pd.DataFrame({'Z': zscore, 'Spread': spread})
    df['returns'] = df['Spread'].pct_change()
    df['position'] = 0
    
    # Logic: Entry at 2.0, Exit at 0.5
    current_pos = 0
    for i in range(len(df)):
        if zscore.iloc[i] <= -2.0: current_pos = 1
        elif zscore.iloc[i] >= 2.0: current_pos = -1
        elif abs(zscore.iloc[i]) <= 0.5: current_pos = 0
        df.iloc[i, df.columns.get_loc('position')] = current_pos

    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    
    # 5. Metrics (The "Quant" edge)
    # Assuming 252 trading days in a year
    annual_return = df['strategy_returns'].mean() * 252
    annual_vol = df['strategy_returns'].std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
    
    print(f"--- Results for {ticker1} & {ticker2} ---")
    print(f"Cointegration p-value: {pvalue:.4f}")
    print(f"Annualized Return: {annual_return:.2%}")
    print(f"Annualized Volatility: {annual_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    return df, pvalue

results_df, p_val = backtest_pairs('XOM', 'CVX', '2023-01-01', '2026-01-01')

import matplotlib.pyplot as plt

# --- Section 6: Plotting the Equity Curve ---
plt.figure(figsize=(12, 6))

# Calculate cumulative returns: (1 + r1)(1 + r2)...
cumulative_returns = (1 + results_df['strategy_returns']).cumprod()

# Plotting the growth of $1
plt.plot(cumulative_returns, label='Pairs Trading Strategy (XOM/CVX)', color='green', linewidth=1.5)
plt.title('Cumulative Returns: XOM vs CVX (Mean Reversion Strategy)')
plt.xlabel('Date')
plt.ylabel('Growth of $1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Section 7: Advanced Risk Metrics ---
# Calculate the rolling peak of the equity curve for drawdown analysis
rolling_max = cumulative_returns.cummax()
drawdown = (cumulative_returns - rolling_max) / rolling_max
max_drawdown = drawdown.min()

# Performance Calculation
annual_return = results_df['strategy_returns'].mean() * 252
calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

print(f"--- Advanced Risk Metrics ---")
print(f"Maximum Drawdown: {max_drawdown:.2%}")
print(f"Calmar Ratio: {calmar_ratio:.2f}")