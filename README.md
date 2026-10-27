# Options-Implied ML Model

An empirical machine learning pipeline that predicts stock returns by combining forward-looking option Greeks and implied volatility levels with put-call parity volatility spreads.

---

## 1. Project Overview

Standard equity prediction models rely heavily on historical stock characteristics, which are backward-looking. This project implements a gradient-boosted decision tree framework alongside a benchmark to forecast stock returns and extreme tail events in U.S. equities.

By merging two foundational options-pricing theories, the model captures both **expected tail magnitude** (via hypothetical at-the-money option implied volatilities and Greeks) and **directional informed order flow** (via deviations from put-call parity).

---

## 2. Theoretical Foundation

### A. Andreou, Han, & Li (2025) — *Predicting Stock Jumps and Crashes Using Options*
* **Core Premise:** Options embed leading information about extreme future price movements (defined as 12-month log returns exceeding +70% for jumps or falling below -70% for crashes).
* **Methodology:** Converts active options (31–90 days to maturity) into hypothetical at-the-money (ATM) equivalents using the Cox-Ross-Rubinstein (CRR) binomial tree model to extract 10 stock-level option features: average call and put implied volatility ($\sigma^c, \sigma^p$), delta ($\Delta^c, \Delta^p$), gamma ($\Gamma^c, \Gamma^p$), theta ($\Theta^c, \Theta^p$), and vega ($V^c, V^p$).
* **Role in Model:** Raw implied volatility and delta serve as primary indicators of impending tail risk and uncertainty, while secondary Greeks (gamma, theta, vega) contribute through non-linear interactions captured by LightGBM. Because raw IV spikes prior to both jumps and crashes, additional directional signals are required to separate positive tails from negative tails.

### B. Cremers & Weinbaum (2010) — *Deviations from Put-Call Parity and Stock Return Predictability*
* **Core Premise:** Grounded in the sequential trade model of Easley, O'Hara, and Srinivas (1998), informed traders often trade in the options market first due to leverage and liquidity. This creates asymmetric price pressure that causes call and put implied volatilities to deviate from strict put-call parity ($IV^{\text{call}} = IV^{\text{put}}$).
* **Methodology:** Pairs call and put options on the same underlying stock with identical strike prices and expiration dates to compute the open-interest-weighted **Volatility Spread** ($VS_{i,t} = IV_{i,t}^{\text{calls}} - IV_{i,t}^{\text{puts}}$) and its recent change ($\Delta VS_{i,t}$).
* **Role in Model:** Provides a directional buying-pressure gauge. Bullish informed trading drives call IV above put IV ($VS > 0$), predicting positive abnormal returns, whereas bearish informed trading drives put IV above call IV ($VS < 0$), predicting negative abnormal returns. Predictability is strongest when option liquidity is high relative to underlying stock liquidity.

---

## 3. Input Features

### Option Features (Andreou-Han-Li)
* **Implied Volatility ($\sigma^c, \sigma^p$):** Average call and put implied volatility estimated via the CRR binomial model.
* **Option Greeks ($\Delta^{c,p}, \Gamma^{c,p}, \Theta^{c,p}, V^{c,p}$):** Average hypothetical ATM call and put delta, gamma, theta, and vega.

### Parity & Informed Flow Features (Cremers-Weinbaum)
* **Volatility Spread Level ($VS_{i,t}$):** Open-interest-weighted average of $(IV^{\text{call}} - IV^{\text{put}})$ across matched strike/maturity pairs.
* **Volatility Spread Change ($\Delta VS_{i,t}$):** Short-term change (e.g., 1-day and 5-day delta) in the volatility spread.
* **Liquidity Interaction Proxies:** Relative option liquidity (bid-ask spread, volume) vs. stock illiquidity (Amihud illiquidity ratio).
---

## 4. Data & Pipeline Specifications
To be determined

---

## 5. Repository Structure

```text
quant-fund-options-ml/
├── data/
│   └── raw/            # Raw option chains, underlying equity data, and market snapshots
├── src/                # Source code (data ingestion, feature engineering, model training)
├── .gitignore          # Git ignore rules
└── README.md           # Project documentation and theoretical foundation
```

### Directory Overview
* **`data/`**: Storage for all dataset inputs and outputs.
  * **`data/raw/`**: Raw, unmodified market data files (historical option chains, underlying equities, quotes).
* **`src/`**: Core source code implementing data ingestion, CRR binomial tree calculations, volatility spread computation, and machine learning models.
* **`.gitignore`**: Specifies untracked files and local environment files (e.g., `.env`, local data caches).
* **`README.md`**: Project overview, theoretical background, feature definitions, and pipeline documentation.

---

## 6. Getting Started

### Dependencies
Install the core environment packages:
```bash
pip install 
```

### Running the Pipeline
To be determined

---

## 7. References

1. **Andreou, P. C., Han, C., & Li, N. (2025).** *Predicting Stock Jumps and Crashes Using Options.* Journal of Futures Markets, 45, 1471–1490.
2. **Cremers, M., & Weinbaum, D. (2010).** *Deviations from Put-Call Parity and Stock Return Predictability.* Journal of Financial and Quantitative Analysis, 45(2), 335–367.
