# Reflection

**Which fault types were hardest to catch, and why?**

1. **Subtle Statistical Shifts (`data_batch` & `ai_infra`)**: Faults that manifest closer to normal operational variance (e.g., deviations around $2.5\sigma - 2.8\sigma$) are the most challenging to detect reliably. Using rigid static baseline bounds ($\mu \pm 3\sigma$) creates a inherent trade-off: tightening the bounds increases False Positive Rate (FPR) on clean variance spikes, while keeping them at $3\sigma$ risks missing subtle distribution shifts, feature skew, or minor embedding centroid drift.
2. **Topological Lineage Anomalies (`lineage_run`)**: Static thresholds provided in `ctx.baseline` only cover runtime durations (`lineage_duration_ms_max`). Structural faults such as `missing_upstream` (a transform graph missing an upstream edge) or `orphan_output` (zero downstream consumers) cannot be caught by static thresholding alone. Solving this required leveraging `ctx.state` to dynamically track and memoize the expected upstream topological sets and downstream counts for each pipeline job across runs.

**What would you change about your cost/coverage tradeoff, if you had another pass?**

Our current architecture prioritizes maximum fault coverage (TPR) by querying metered inspection tools across incoming events. On the Practice stream, this achieved a perfect **50.0/50.0 score** ($\text{TPR}=100\%, \text{FPR}=0\%, \text{cost}=180/220$). On the larger Public stream (160 events), calling tools on all events resulted in a slight budget overage ($\text{cost}=240/220$), achieving a score of **43.84/50.0** with `high` bands across all four pillars.

Analyzing the scoring formula $\text{Score} = 100 \times (0.5 \cdot \text{TPR} - 0.3 \cdot \text{FPR} - 0.2 \cdot \min(\text{cost\_overage}, 1))$ reveals why prioritizing coverage over minor cost overage is mathematically optimal:
* In a stream with ~40 faulty events, catching 1 additional fault increases TPR by $2.5\%$, adding **+1.25 points** to the score.
* Exceeding the 220 credit budget by an average tool call ($1.5$ credits) increases `cost_overage` by $0.68\%$, penalizing the score by only **-0.136 points**.
* Therefore, cutting off tool calls to save budget when credits run low is a losing strategy if it causes even a single missed fault.

If we had another pass to further optimize the trade-off, we would implement:
1. **Pre-RPC Payload Triage**: Exploit free metadata inside `payload` (e.g., `declared_sla`, `columns`, `producer`) before spending credits on `ctx.tools` RPCs.
2. **Stateful Dynamic Baselines**: Use `ctx.state` to maintain Welford running statistics or Exponential Moving Averages (EMA) on clean batches. This allows detecting progressive feature drift or staleness lag with higher statistical confidence without inflating FPR.
