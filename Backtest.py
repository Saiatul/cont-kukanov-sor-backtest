import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from itertools import product

# Load and deduplicate data
df = pd.read_csv("l1_day.csv")
df_cleaned = df.sort_values("ts_event").drop_duplicates(subset=["ts_event", "publisher_id"])

# Compute cost of a given allocation
def compute_cost(split, venues, order_size, lambda_over, lambda_under, theta_queue):
    executed = 0
    cash_spent = 0
    for i, alloc in enumerate(split):
        ask = venues.iloc[i]['ask_px_00']
        fee = 0  # Assuming fee = 0
        rebate = 0  # Assuming rebate = 0
        ask_size = venues.iloc[i]['ask_sz_00']

        exe = min(alloc, ask_size)
        executed += exe
        cash_spent += exe * (ask + fee)
        rebate_cash = max(alloc - exe, 0) * rebate
        cash_spent -= rebate_cash

    underfill = max(order_size - executed, 0)
    overfill = max(executed - order_size, 0)
    cost_penalty = lambda_under * underfill + lambda_over * overfill
    risk_penalty = theta_queue * (underfill + overfill)

    return cash_spent + cost_penalty + risk_penalty

# Brute-force allocator per snapshot
def allocator(order_size, venues, lambda_over, lambda_under, theta_queue):
    step = 500
    N = len(venues)
    splits = [[]]

    for v in range(N):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size - used, venues.iloc[v]['ask_sz_00'])
            for q in range(0, int(max_v) + 1, step):
                new_splits.append(alloc + [q])
        splits = new_splits

    best_cost = float('inf')
    best_split = None

    for alloc in splits:
        if sum(alloc) != order_size:
            continue
        cost = compute_cost(alloc, venues, order_size, lambda_over, lambda_under, theta_queue)
        if cost < best_cost:
            best_cost = cost
            best_split = alloc

    # Fallback in case no valid split found
    if best_split is None:
        # Simple proportional fallback
        total_size = venues["ask_sz_00"].sum()
        if total_size == 0:
            return [0] * len(venues), 0
        fallback_split = []
        for _, row in venues.iterrows():
            alloc = min((row["ask_sz_00"] / total_size) * order_size, row["ask_sz_00"])
            fallback_split.append(int(alloc // step) * step)  # round down to step
        return fallback_split, compute_cost(fallback_split, venues, order_size, lambda_over, lambda_under, theta_queue)

    return best_split, best_cost


# Backtest using allocator
def backtest_allocator(df_snapshots, order_size, lambda_over, lambda_under, theta_queue):
    remaining = order_size
    total_cost = 0
    cumulative_costs = []

    for ts, snapshot in df_snapshots.groupby("ts_event"):
        if remaining <= 0:
            break

        venues = snapshot[["publisher_id", "ask_px_00", "ask_sz_00"]].copy().reset_index(drop=True)

        alloc_split, _ = allocator(remaining, venues, lambda_over, lambda_under, theta_queue)

        for i, qty in enumerate(alloc_split):
            ask_price = venues.iloc[i]["ask_px_00"]
            ask_size = venues.iloc[i]["ask_sz_00"]
            fill_qty = min(qty, ask_size, remaining)
            cost = fill_qty * ask_price
            total_cost += cost
            remaining -= fill_qty

        cumulative_costs.append(total_cost)

    avg_price = total_cost / (order_size - remaining) if (order_size - remaining) > 0 else 0
    return total_cost, avg_price, cumulative_costs

# Best-Ask Baseline
def backtest_best_ask(df_snapshots, order_size):
    remaining = order_size
    total_cost = 0
    cumulative_costs = []

    for ts, snapshot in df_snapshots.groupby("ts_event"):
        if remaining <= 0:
            break
        best_row = snapshot.loc[snapshot["ask_px_00"].idxmin()]
        ask_price = best_row["ask_px_00"]
        ask_size = best_row["ask_sz_00"]
        fill_qty = min(ask_size, remaining)
        total_cost += fill_qty * ask_price
        remaining -= fill_qty
        cumulative_costs.append(total_cost)

    avg_price = total_cost / (order_size - remaining) if (order_size - remaining) > 0 else 0
    return total_cost, avg_price, cumulative_costs

# TWAP Baseline
def backtest_twap(df_snapshots, order_size):
    snapshots_count = df_snapshots["ts_event"].nunique()
    per_snapshot = order_size / snapshots_count
    total_cost = 0
    cumulative_costs = []
    remaining = order_size

    for ts, snapshot in df_snapshots.groupby("ts_event"):
        if remaining <= 0:
            break
        qty_to_buy = min(per_snapshot, remaining)
        total_available = snapshot["ask_sz_00"].sum()
        if total_available == 0:
            cumulative_costs.append(total_cost)
            continue

        for _, row in snapshot.iterrows():
            weight = row["ask_sz_00"] / total_available
            alloc_qty = weight * qty_to_buy
            fill_qty = min(alloc_qty, row["ask_sz_00"], remaining)
            cost = fill_qty * row["ask_px_00"]
            total_cost += cost
            remaining -= fill_qty
        cumulative_costs.append(total_cost)

    avg_price = total_cost / (order_size - remaining) if (order_size - remaining) > 0 else 0
    return total_cost, avg_price, cumulative_costs

# VWAP Baseline
def backtest_vwap(df_snapshots, order_size):
    remaining = order_size
    total_cost = 0
    cumulative_costs = []

    for ts, snapshot in df_snapshots.groupby("ts_event"):
        if remaining <= 0:
            break
        total_available = snapshot["ask_sz_00"].sum()
        if total_available == 0:
            cumulative_costs.append(total_cost)
            continue

        for _, row in snapshot.iterrows():
            weight = row["ask_sz_00"] / total_available
            alloc_qty = weight * remaining
            fill_qty = min(alloc_qty, row["ask_sz_00"], remaining)
            cost = fill_qty * row["ask_px_00"]
            total_cost += cost
            remaining -= fill_qty
        cumulative_costs.append(total_cost)

    avg_price = total_cost / (order_size - remaining) if (order_size - remaining) > 0 else 0
    return total_cost, avg_price, cumulative_costs

# BPS calculation
def bps(saved, baseline):
    return ((baseline - saved) / baseline) * 10000

# Parameter grid search
lambda_over_values = [0.1, 0.3, 0.5, 0.7, 1.0]
lambda_under_values = [0.1, 0.3, 0.5, 0.7, 1.0]
theta_queue_values = [5, 10, 15, 20, 30]


best_result = None
best_params = (10,10,1)

for lo in lambda_over_values:
    for lu in lambda_under_values:
        for tq in theta_queue_values:
            total_cost, avg_price, _ = backtest_allocator(df_cleaned, 5000, lo, lu, tq)
            if best_result is None or total_cost < best_result:
                best_result = total_cost
                best_params = (lo, lu, tq)

# Final run with best params
allocator_cost, allocator_avg, allocator_cum = backtest_allocator(df_cleaned, 5000, *best_params)
bestask_cost, bestask_avg, bestask_cum = backtest_best_ask(df_cleaned, 5000)
twap_cost, twap_avg, twap_cum = backtest_twap(df_cleaned, 5000)
vwap_cost, vwap_avg, vwap_cum = backtest_vwap(df_cleaned, 5000)

# Final JSON result
result = {
    "best_parameters": {
        "lambda_over": best_params[0],
        "lambda_under": best_params[1],
        "theta_queue": best_params[2]
    },
    "allocator": {
        "total_cash_spent": allocator_cost,
        "average_fill_price": allocator_avg
    },
    "best_ask": {
        "total_cash_spent": bestask_cost,
        "average_fill_price": bestask_avg
    },
    "twap": {
        "total_cash_spent": twap_cost,
        "average_fill_price": twap_avg
    },
    "vwap": {
        "total_cash_spent": vwap_cost,
        "average_fill_price": vwap_avg
    },
    "savings_vs_best_ask_bps": bps(allocator_cost, bestask_cost),
    "savings_vs_twap_bps": bps(allocator_cost, twap_cost),
    "savings_vs_vwap_bps": bps(allocator_cost, vwap_cost)
}

print(json.dumps(result, indent=4))

# plot

# Convert to NumPy arrays
allocator_cum = np.array(allocator_cum)
bestask_cum = np.array(bestask_cum)
twap_cum = np.array(twap_cum)
vwap_cum = np.array(vwap_cum)

# Normalize inline
allocator_norm = (allocator_cum - allocator_cum[0]) / (allocator_cum[-1] - allocator_cum[0]) if allocator_cum[-1] != allocator_cum[0] else allocator_cum
bestask_norm = (bestask_cum - bestask_cum[0]) / (bestask_cum[-1] - bestask_cum[0]) if bestask_cum[-1] != bestask_cum[0] else bestask_cum
twap_norm = (twap_cum - twap_cum[0]) / (twap_cum[-1] - twap_cum[0]) if twap_cum[-1] != twap_cum[0] else twap_cum
vwap_norm = (vwap_cum - vwap_cum[0]) / (vwap_cum[-1] - vwap_cum[0]) if vwap_cum[-1] != vwap_cum[0] else vwap_cum

# Resample all to 25 evenly spaced points
allocator_resampled = np.interp(np.linspace(0, len(allocator_norm)-1, 25), np.arange(len(allocator_norm)), allocator_norm)
bestask_resampled = np.interp(np.linspace(0, len(bestask_norm)-1, 25), np.arange(len(bestask_norm)), bestask_norm)
twap_resampled = np.interp(np.linspace(0, len(twap_norm)-1, 25), np.arange(len(twap_norm)), twap_norm)
vwap_resampled = np.interp(np.linspace(0, len(vwap_norm)-1, 25), np.arange(len(vwap_norm)), vwap_norm)

# Plot
x_axis = np.arange(25)
plt.figure(figsize=(10, 6))
plt.plot(x_axis, allocator_resampled, label='Allocator')
plt.plot(x_axis, bestask_resampled, label='Best Ask')
plt.plot(x_axis, twap_resampled, label='TWAP')
plt.plot(x_axis, vwap_resampled, label='VWAP')
plt.title('Normalized Cumulative Cost Over Time')
plt.xlabel('Snapshot Index (0–24)')
plt.ylabel('Normalized Cumulative Cost')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results.png')
plt.show()

# Note: VWAP and Best Ask produce very similar costs, so their lines overlap in the plot.