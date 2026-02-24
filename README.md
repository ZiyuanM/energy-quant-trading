# energy-quant-trading
A statistical arbitrage strategy using cointegration to trade US energy pairs (XOM/CVX).
# Energy Sector Quantitative Research Lab
### Statistical Arbitrage & Mean Reversion in US Energy Majors

## Project Overview
This repository contains a quantitative framework for identifying and trading cointegrated pairs within the US Energy sector. The project evolved through three research phases:

1. **Phase 1: Proof of Concept**: Established a base mean-reversion model for XOM/CVX.
2. **Phase 2: Risk Management**: Integrated a rolling volatility filter and a 4-sigma Stop-Loss to manage structural decoupling risks.

## Key Research Findings
* **Strongest Sector Tether**: Chevron (CVX) showed the most consistent cointegration across the sector.
* **The "Volatility Guard"**: Implementing a volatility gate successfully prevented entries during the 2024 decoupling of CVX/VLO.

## Technical Stack
- **Data**: Yahoo Finance API
- **Statistics**: Statsmodels (OLS, Engle-Granger Cointegration)
- **Visualization**: Seaborn Heatmaps & Matplotlib Equity Curves
