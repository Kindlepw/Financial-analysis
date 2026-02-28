# Financial-analysis

A Python tool that builds a **Discounted Cash Flow (DCF) valuation model** for every company in the S&P 500 index, using publicly available financial data from Yahoo Finance.

## Features

- **Automated data sourcing** – Fetches the current S&P 500 ticker list from Wikipedia and retrieves historical free cash flow (FCF), stock prices, beta, capital structure, and other financials from Yahoo Finance.
- **DCF modelling** – Projects FCF over a 5–10 year horizon with linearly fading growth rates, estimates terminal value using both the **Gordon Growth Model** (perpetuity growth) and an **exit multiple** method (blended average), and discounts all cash flows at the company's **WACC**.
- **WACC estimation** – Derives cost of equity via CAPM (risk-free rate + beta × ERP) and after-tax cost of debt from the income statement, weighted by market-value capital structure.
- **Valuation signal** – Compares intrinsic value per share to the current market price and assigns a **Buy / Hold / Sell** signal (≥ +15 % upside → Buy; ≥ −15 % downside → Sell).
- **Structured output** – Produces a sorted table and saves results to a CSV file with columns: Ticker, Company, Sector, Current Price, Intrinsic Value, % Upside/Downside, Valuation Signal, WACC, FCF Growth Rate.

## Project structure

```
Financial-analysis/
├── main.py               # CLI entry point
├── requirements.txt      # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py   # S&P 500 ticker list + Yahoo Finance data retrieval
│   ├── dcf_model.py      # DCF model (WACC, FCF projection, terminal value)
│   └── valuation.py      # Valuation engine + signal generation + reporting
└── tests/
    ├── test_data_fetcher.py
    ├── test_dcf_model.py
    └── test_valuation.py
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full S&P 500 analysis

```bash
python main.py
```

Results are saved to `sp500_dcf_valuation.csv` in the current directory.

### 3. Analyse a subset of tickers

```bash
python main.py --tickers AAPL MSFT GOOGL AMZN META
```

### 4. Custom output path and verbose logging

```bash
python main.py --output results/valuation.csv --verbose
```

### 5. All CLI options

```
usage: main.py [-h] [--tickers TICKER [TICKER ...]] [--output PATH]
               [--delay SECONDS] [--top N] [--verbose]

optional arguments:
  --tickers   Space-separated ticker symbols (default: full S&P 500 list)
  --output    Output CSV file path (default: sp500_dcf_valuation.csv)
  --delay     Seconds between API calls to avoid rate-limiting (default: 0.5)
  --top N     Number of top Buy/Sell companies to highlight (default: 20)
  --verbose   Enable DEBUG-level logging
```

## Example output

```
================================================================================
S&P 500 DCF VALUATION SUMMARY
================================================================================
Total companies analysed : 480
  Buy  (≥+15% upside)     : 142
  Hold (−15% to +15%)     : 198
  Sell (≥−15% downside)   : 140
================================================================================

Top 5 Most Undervalued (BUY signal):
Ticker  Company          Current Price ($)  Intrinsic Value ($)  % Upside/Downside  Valuation Signal
...
```

The full table is also written to the CSV file with all nine columns.

## Methodology notes

| Parameter | Value / Source |
|---|---|
| Risk-free rate | ~4.5 % (10-year US Treasury proxy) |
| Equity risk premium | 5.5 % (historical US ERP) |
| Terminal growth rate | 2.5 % (long-run nominal GDP) |
| FCF exit multiple | 20× (terminal year FCF) |
| Projection horizon | 7 years (linear growth fade) |
| Buy threshold | ≥ +15 % upside |
| Sell threshold | ≥ −15 % downside |

> **Disclaimer**: This tool is for educational and research purposes only.  
> DCF valuations are highly sensitive to input assumptions and should not be  
> used as the sole basis for investment decisions.

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```

