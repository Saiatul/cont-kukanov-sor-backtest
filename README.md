# Cont & Kukanov Smart Order Router Backtest

This project implements and evaluates a Smart Order Router (SOR) based on the cost optimization framework by **Cont & Kukanov**, designed for optimal order placement in fragmented limit order markets. The goal is to minimize the execution cost of a **5,000-share buy order** by optimally splitting it across multiple trading venues, considering overfill risk, underfill penalties, and queue position risk.

---

## 📘 Background

Cont & Kukanov's model introduces a **static cost function** that penalizes overfilling, underfilling, and the risk of not getting filled due to queue position. The Smart Order Router makes allocation decisions at each snapshot of the market and aims to complete the order by the end of the data window.

The router is benchmarked against three baseline strategies:
- **Best Ask** – always hits the cheapest available price
- **TWAP (Time-Weighted Average Price)** – divides the order evenly over time
- **VWAP (Volume-Weighted Average Price)** – allocates based on available volume

---

## 📂 Files

- `Backtest.py` – Main Python script that implements the allocator, baselines, parameter search, and cumulative cost plot
- `l1_day.csv` – Market snapshot data from Aug 1, 2024 (provided)
- `results.png` – Normalized cumulative cost comparison plot for Allocator vs Baselines
- `README.md` – This documentation

---

## ⚙️ Methodology

### 1. **Data Cleaning**
- The data stream is preprocessed by grouping on `ts_event` and retaining only the first message per `publisher_id`.
- This results in one clean Level-1 snapshot per timestamp, across venues.

### 2. **Allocator Logic**
The allocator:
- Takes in the current market snapshot and the remaining order.
- Explores all feasible allocations in steps (default: `step = 500`).
- Evaluates each allocation using the Cont & Kukanov cost model:
- If no valid allocation is found, a **smart fallback** is triggered:
- Sorts venues by best price and size
- Greedily allocates shares in descending order of quality

### 3. **Baselines**
Three industry-standard strategies were implemented:
- **Best Ask**: Buys from the venue with the lowest ask price
- **TWAP**: Buys evenly over each snapshot
- **VWAP**: Allocates based on available size at each venue

---

## Parameter Search

A grid search is used to find the optimal combination of risk penalties:

- `lambda_over`: [0.1, 0.3, 0.5, 0.7, 1.0]
- `lambda_under`: [0.1, 0.3, 0.5, 0.7, 1.0]
- `theta_queue`: [5, 10, 15, 20, 30]


This setup achieved strong bps savings while maintaining a quick execution time (~14.5s).

---

## 📊 Results

| Strategy    | Total Cost   | Avg Fill Price | BPS vs Allocator |
|-------------|--------------|----------------|------------------|
| **Allocator** | \$1,114,100  | 222.740         |               |
| Best Ask    | \$1,114,102.28 | 222.820         | +3.61 bps        |
| TWAP        | \$1,115,309.31 | 223.061         | +14.43 bps       |
| VWAP        | \$1,114,102.28 | 222.820         | +3.61 bps        |

**Key Takeaways**:
- Allocator outperformed **Best Ask by 3.61 bps** and **TWAP by 14.43 bps**
- Plot `results.png` confirms more cost-efficient execution over time
- The allocator executed conservatively and efficiently with high `theta_queue`

---

## Improvement Suggestion

To enhance realism in future versions:
- Introduce **slippage modeling** based on book pressure
- Dynamically adjust penalties based on market volatility
- Model **queue depletion probabilities** or latency risk

---

## How to Run

Ensure Python 3.8+ with `pandas`, `numpy`, and `matplotlib` installed.

```bash
python Backtest.py
