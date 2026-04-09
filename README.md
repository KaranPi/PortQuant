# PortQuant: Portfolio Quantitative Analysis Framework

## 📊 Overview

PortQuant is a comprehensive Python-based quantitative analysis framework for portfolio management, risk analysis, and statistical modeling of financial assets. Built with Jupyter notebooks and a modular library architecture, it provides sophisticated tools for analyzing correlations, volatility dynamics, value-at-risk metrics, and portfolio performance optimization.

## ✨ Core Features

- **Data Ingestion & Preparation**: Automated pipelines for importing and preprocessing financial time-series data from multiple sources
- **Returns Analysis**: Comprehensive calculation and analysis of asset returns across multiple time horizons (daily, monthly, quarterly, yearly)
- **Statistical Analysis**: Descriptive statistics, normality testing, and distribution fitting
- **Distribution Modeling**: Fit returns to normal, Student-t, and other parametric distributions with visualization
- **Correlation & Beta Analysis**: Dynamic correlation matrices, rolling beta calculations, and factor exposure analysis
- **Portfolio Analytics**: Portfolio construction, weight allocation, performance tracking, and drawdown analysis
- **Volatility Forecasting**: EWMA (Exponentially Weighted Moving Average) and GARCH(1,1) volatility models with comparative analysis
- **Value-at-Risk (VaR) & Expected Shortfall (ES)**: Historical, parametric (normal), Student-t, and GARCH-based risk measures at multiple confidence levels
- **Professional Visualizations**: Publication-ready charts and heatmaps for all analysis outputs

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/KaranPi/PortQuant.git
cd PortQuant
```

Create and activate the conda environment:
```bash
conda env create -f environment_minimal.yml
conda activate quantlite
```

Configure paths in config.yaml

paths:
```
  data_raw: /path/to/raw/data
  data_int: /path/to/intermediate/data
  figures: /path/to/figures/output
```

Dependencies
```
Python 3.9.5
numpy 1.19.5 - Numerical computing
pandas 1.2.4 - Data manipulation and analysis
scipy 1.7.0 - Scientific computing
statsmodels 0.13.5 - Statistical modeling and hypothesis tests
scikit-learn 0.24.2 - Machine learning and preprocessing
numba 0.53.1 - JIT compilation for performance
matplotlib 3.3.4 - Visualization
jupyterlab 3.0.14 - Interactive analysis environment
```

📁 Project Structure
Code
```
PortQuant/
├── notebooks/                          # Sequential analysis workflow
│   ├── 01_ingest.ipynb                # Data ingestion from multiple sources
│   ├── 01a_ingest_jugaad.ipynb        # Alternative data ingestion pipeline
│   ├── 01a_ingest_jugaad_new.ipynb    # Enhanced jugaad ingestion
│   ├── 01.x_ingest_and_prep.ipynb     # Combined ingestion and preparation
│   ├── 02_returns.ipynb               # Return calculations and analysis
│   ├── 03_stats.ipynb                 # Descriptive statistics and tests
│   ├── 04_distributions.ipynb         # Distribution fitting and analysis
│   ├── 05_corr_beta.ipynb             # Correlation and beta analysis
│   ├── 06_portfolio.ipynb             # Portfolio construction and analytics
│   ├── 07_ewma.ipynb                  # EWMA volatility modeling
│   └── 08_var_es.ipynb                # VaR and Expected Shortfall analysis
├── src/quantlib/                       # Core quantitative library
│   ├── __init__.py                    # Package initialization
│   ├── io.py                          # Data ingestion and I/O operations
│   ├── returns.py                     # Returns calculations
│   ├── stats.py                       # Statistical functions and tests
│   ├── risk.py                        # Risk metrics (VaR, ES, GARCH)
│   ├── factors.py                     # Factor-based analysis
│   └── viz.py                         # Visualization utilities
├── figures/                            # Output figures and visualizations
├── data_raw/                           # Raw input data directory
├── data_int/                           # Intermediate processed data
├── config.yaml                         # Configuration parameters
├── environment_minimal.yml             # Conda environment specification
└── README.md                           # This file
```

## ⚙️ Configuration

The `config.yaml` file controls all analysis parameters:

```yaml
ingest:
  anchor_symbol: HDFCBANK              # Anchor stock for data alignment

windows:
  short: 5                             # 5 trading days
  month: 21                            # ~1 month of trading days
  quarter: 63                          # ~1 quarter
  year: 252                            # Annual trading days

risk:
  lambdas: [0.94, 0.97]                # EWMA decay parameters
  var_alphas: [0.95, 0.99]             # VaR confidence levels
  hist_windows: [252, 1260]            # Historical windows (1yr, 5yr)
  t_fit_window: 252                    # Student-t fitting window
  future_days: 30                      # Forward-looking horizon
  ewma_forward_windows: [2, 5, 21, 63, 252]  # Multi-horizon EWMA

portfolio:
  notional_inr: 100000                 # Portfolio notional value
  weights_file: data_int/portfolio_weights.csv
```

📊 Proof-of-Work Gallery
Correlation Analysis
Daily Correlation Matrix ![Daily Correlation](figures/corr_daily.png)

63-Day Rolling Correlation Heatmap ![63-Day Correlation Heatmap](figures/corr_heatmap_63d_2025-09-09.png)

Daily Correlation Heatmap ![Daily Correlation Heatmap](figures/corr_heatmap_daily.png)

Volatility & Distribution Analysis
Volatility Comparison (EWMA vs GARCH) ![Volatility Overlay](figures/vol_overlay_ewma_garch_21d.png)

Sigma Comparison Across Assets ![Sigma Comparison](figures/sigma_cmp.png)

Individual Asset Distribution Fits

AEROFLEX PDF: ![AEROFLEX Distribution](figures/pdf_AEROFLEX.png)
BAJFINANCE PDF: ![BAJFINANCE Distribution](figures/pdf_BAJFINANCE.png)
IRCTC PDF: ![IRCTC Distribution](figures/pdf_IRCTC.png)
MCX PDF: ![MCX Distribution](figures/pdf_MCX.png)
NEWGEN PDF: ![NEWGEN Distribution](figures/pdf_NEWGEN.png)
ZOMATO PDF: ![ZOMATO Distribution](figures/pdf_ZOMATO.png)
Value-at-Risk (VaR) Analysis
VaR 95% Confidence - Historical vs Normal Distribution ![VaR 95% Historical vs Normal](figures/var_bands_95_hist252_vs_norm_l94.png)

VaR 99% Confidence - Historical vs Normal Distribution ![VaR 99% Historical vs Normal](figures/var_bands_99_hist1260_vs_norm_l97.png)

VaR 95% - Student-t Distribution (252-day window) ![VaR 95% Student-t 252d](figures/var_bands_t_95_252.png)

VaR 95% - Student-t Distribution (63-day window) ![VaR 95% Student-t 63d](figures/var_bands_t_95_63.png)

VaR 99% - Student-t Distribution (252-day window) ![VaR 99% Student-t 252d](figures/var_bands_t_99_252.png)

VaR 99% - Student-t Distribution (63-day window) ![VaR 99% Student-t 63d](figures/var_bands_t_99_63.png)

Historical VaR 95% ![Historical VaR 95%](figures/var_hist_95.png)

Multi-Horizon Risk Analysis
VaR/ES Horizon Analysis - EWMA ![VaR/ES EWMA Horizon](figures/var_es_horizon_ewma.png)

VaR/ES Horizon Analysis - GARCH(1,1) ![VaR/ES GARCH Horizon](figures/var_es_horizon_garch11.png)

VaR/ES Horizon Analysis - Standard GARCH ![VaR/ES GARCH Horizon](figures/var_es_horizon_garch.png)

Portfolio Analytics
Portfolio Weight Allocation ![Portfolio Weights](figures/weights.png)

Portfolio Risk Contribution (252-day window) ![Portfolio Risk Contribution](figures/portfolio_risk_contrib_252.png)

Portfolio Value & Drawdown Analysis ![Portfolio Drawdown](figures/portfolio_value_dd.png)

🔄 Analysis Workflow
The framework follows a sequential workflow through 8 notebooks:
```
Data Ingestion (01, 01a, 01.x) - Import from CSV, APIs, or data providers
Returns Calculation (02) - Compute log returns and analyze price changes
Statistical Testing (03) - Normality tests, descriptive statistics
Distribution Fitting (04) - Fit parametric distributions, test goodness-of-fit
Correlation & Beta (05) - Rolling correlations, factor exposures
Portfolio Construction (06) - Allocate weights, track performance
Volatility Modeling (07) - EWMA and GARCH(1,1) implementations
Risk Measurement (08) - VaR, ES across multiple methodologies
```

💡 Key Capabilities
```
Risk Metrics
Value-at-Risk (VaR): Parametric, historical, Student-t, and GARCH-based methods
Expected Shortfall (ES): Conditional risk measure at multiple confidence levels
Volatility Forecasting: EWMA with configurable decay parameters
GARCH(1,1) Modeling: Mean-reverting volatility dynamics
Risk Contribution: Marginal and component VaR analysis
Statistical Methods
Returns normality testing (Jarque-Bera, Shapiro-Wilk)
Distribution fitting (Normal, Student-t, empirical)
Autocorrelation analysis
Rolling statistics across multiple windows
Portfolio Analysis
Multi-asset correlation matrices
Beta and factor exposure calculations
Portfolio performance metrics
Drawdown and recovery analysis
Risk decomposition by asset
```
📝 Usage Example
```Python
import pandas as pd
from src.quantlib import io, returns, risk, viz

# Load configuration
config = io.load_config('config.yaml')

# Ingest data
prices = io.load_prices(config['paths']['data_raw'])

# Calculate returns
rets = returns.log_returns(prices)

# Compute VaR
var_95 = risk.var_parametric(rets, alpha=0.95)
var_99 = risk.var_parametric(rets, alpha=0.99)

# Visualize results
viz.plot_returns(rets)
```

🛠️ Core Modules Documentation
```
io.py
Handles data ingestion, loading configurations, and I/O operations from CSV, APIs, and databases.

returns.py
Computes various return metrics: log returns, simple returns, cumulative returns, and return statistics.

stats.py
Provides statistical functions: descriptive statistics, normality tests, distribution fitting, and hypothesis testing.

risk.py
Comprehensive risk measurement including VaR (multiple methodologies), Expected Shortfall, EWMA and GARCH volatility models.

factors.py
Factor-based analysis tools for beta estimation and factor exposure decomposition.

viz.py
Advanced visualization utilities for creating publication-quality charts, heatmaps, and time-series plots.
```

📈 Use Cases
```
Risk Management: Monitor portfolio VaR and stress-test against historical scenarios
Asset Allocation: Optimize weights based on correlation and volatility forecasts
Performance Attribution: Decompose returns by factor exposures
Regulatory Reporting: Generate VaR reports for compliance (Basel III, etc.)
Research: Analyze distributional properties and volatility dynamics
Trading: Develop volatility-aware trading strategies using GARCH forecasts
```
