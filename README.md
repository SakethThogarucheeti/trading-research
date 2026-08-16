# trading-research

CLI for stock universe selection, feature computation, and strategy backtesting/walk-forward/Monte Carlo research. A thin, deterministic consumer of [trading-platform](https://github.com/SakethThogarucheeti/trading-platform) and the strategy/risk SDKs — not a home for strategy or risk-filter content itself.

See [trading-research-FLOW.md](../trading-research-FLOW.md) for the full research flow this CLI implements, and [trading-research-PLAN.md](../trading-research-PLAN.md) for the build plan and milestone status.

## Status

M1 in progress: promoting the backtest/walk-forward/Monte Carlo engine (originally built in `trading-integ-tests`) into this repo, unchanged apart from import paths. `research/session.py`, `report_base.py`, `registry.py`, `backtesting/`, `walk_forward/`, `monte_carlo/`, `simulators/` are promoted. CLI wiring and universe/feature stages (M2+) are not yet built.

## A note on dependencies

`trading-research` depends on `trading-platform` as a pinned package (`git+...`, tagged), same pattern as `trading-strategy-sdk` and `trading-risk-sdk`. The promoted backtest engine's execution wiring (`BacktestSession`, driving `SignalGenerator` → `RiskFilter` → `OrderExecutor`) imports `trading-platform` internals directly (`trading.strategy.service.generator`, `trading.risk.gates.*`, `trading.execution.service.*`) rather than going through `trading-strategy-sdk`'s `create_strategy()` or `trading-risk-sdk`'s `create_gate()` alone — those SDK factories construct *strategy/risk-gate content* by name, but the pipeline orchestration classes that actually run a backtest (`SignalGenerator`, `RiskFilter`, `OrderExecutor`, `PositionAccountant`) are pipeline logic that lives in `trading-platform`, not in either SDK, and were never meant to be. `trading-platform`'s Python API is treated as `trading-research`'s real contract, pinned the same way as everything else — this is a deliberate choice, not a shortcut to revisit later.

## Reports

Every research session (backtest, walk-forward, Monte Carlo, and future stages) writes a `SessionReport` to `results_dir/{session_id}/report.json` (+ `.html`) — the same contract `trading-platform`'s `/api/reports/*` endpoints already serve, so every session is automatically visible on `trading-dashboard` with no backend changes needed.
