# %%
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint

# --- CONFIGURATION ---
# Active US/Global Energy companies (Updated to avoid delisted HES/PXD)
TICKERS = ['XOM', 'CVX', 'BP', 'SHEL', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'OXY', 'PSX']
START_DATE = '2023-01-01'
END_DATE = '2026-01-01'

def get_robust_data(tickers, start, end):
    print(f"Downloading data for: {tickers}")
    data = yf.download(tickers, start=start, end=end)
    
    # Handle multi-index columns from yfinance
    if 'Adj Close' in data.columns:
        df = data['Adj Close']
    else:
        df = data['Close']
    
    clean_df = df.dropna(axis=1)
    return clean_df

def find_cointegrated_pairs(data):
    n = data.shape[1]
    pvalue_matrix = np.ones((n, n))
    keys = data.columns
    pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            pvalue = result[1]
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j], pvalue))
    
    return pvalue_matrix, pairs

def backtest_pair_v3(data, t1, t2):
    S1 = data[t1]
    S2 = data[t2]
    
    # Calculate Hedge Ratio (Beta)
    S1_with_const = sm.add_constant(S1)
    results = sm.OLS(S2, S1_with_const).fit()
    beta = results.params[t1]
    
    # Calculate Spread and Z-Score
    spread = S2 - beta * S1
    zscore = (spread - spread.mean()) / np.std(spread)
    
    df = pd.DataFrame({'Z': zscore, 'Spread_Return': spread.pct_change()})
    
    # --- V3 RISK MANAGEMENT LAYER ---
    df['rolling_vol'] = df['Z'].rolling(window=20).std()
    df['position'] = 0
    
    current_pos = 0
    for i in range(1, len(df)):
        z = df['Z'].iloc[i]
        vol = df['rolling_vol'].iloc[i]

        # 1. Entry Logic with Volatility Filter
        # Avoid entering when the spread is too "wild" (vol > 1.25)
        if current_pos == 0:
            if z <= -2.0 and vol < 1.25:
                current_pos = 1
            elif z >= 2.0 and vol < 1.25:
                current_pos = -1

        # 2. Hard Stop-Loss (Circuit Breaker)
        # If Z-score hits 4.0, the pair is fundamentally broken. Exit immediately.
        elif abs(z) >= 4.0:
            current_pos = 0

        # 3. Standard Exit Logic
        # Exit when the spread returns toward the mean
        elif (current_pos == 1 and z >= -0.5) or (current_pos == -1 and z <= 0.5):
            current_pos = 0
            
        df.iloc[i, df.columns.get_loc('position')] = current_pos
        
    # Shift position by 1 day to avoid look-ahead bias (trading on tomorrow's price)
    df['strategy_returns'] = df['position'].shift(1) * df['Spread_Return']
    return df

# --- MAIN EXECUTION ---
prices = get_robust_data(TICKERS, START_DATE, END_DATE)

if not prices.empty:
    pvalues, coint_pairs = find_cointegrated_pairs(prices)

    # 1. Visualize Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(pvalues, xticklabels=prices.columns, yticklabels=prices.columns, 
                cmap='RdYlGn_r', mask=(pvalues > 0.05), annot=True, fmt=".2f",
                vmin=0, vmax=0.05)
    plt.title('Energy Sector Cointegration v3.0 (Significant Pairs Only)')
    plt.show()

    # 2. Backtest Best Pair with Risk Controls
    if coint_pairs:
        coint_pairs.sort(key=lambda x: x[2])
        t1, t2, p = coint_pairs[0]
        
        print(f"\nBacktesting Best Pair: {t1} & {t2} (p={p:.4f})")
        results = backtest_pair_v3(prices, t1, t2)
        
        # Performance Visualization
        cum_rets = (1 + results['strategy_returns'].fillna(0)).cumprod()
        plt.figure(figsize=(12, 6))
        plt.plot(cum_rets, color='navy', label=f'Strategy: {t1}/{t2}')
        plt.title('Equity Curve v3.0 (Including Stop-Loss & Vol Filter)')
        plt.ylabel('Growth of $1')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Risk Metrics
        ann_ret = results['strategy_returns'].mean() * 252
        ann_vol = results['strategy_returns'].std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
        
        # Max Drawdown Calculation
        rolling_max = cum_rets.cummax()
        drawdown = (cum_rets - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        print(f"Annualized Return: {ann_ret:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.2%}")
    else:
        print("No significant pairs found.")
# %%
