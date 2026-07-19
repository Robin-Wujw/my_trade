# FinHack and MiniQMT study notes

Date: 2026-07-18

Branch context: this note was created on `codex/study-finhack-miniqmt`, branched from the current MiniQMT worktree. The local invariant is that MiniQMT remains a read-only data/account adapter plus a backtest execution profile. Live order placement is still disabled.

External source inspected:

- Repository: https://github.com/FinHackCN/finhack
- Shallow clone commit: `dedbbd0`
- Files reviewed: `README.md`, `setup.py`, `finhack/core/core.py`, `finhack/core/command/finhack.py`, `finhack/core/loader/base_loader.py`, `finhack/library/class_loader.py`, `finhack/factor/default/factorManager.py`, `finhack/market/astock/astock.py`, `finhack/widgets/templates/empty_project/trader/sim/qmt_trader.py`, `finhack/widgets/templates/empty_project/trader/sim/rules.py`, `examples/demo-project/strategy/QMTStrategy.py`.

## What FinHack is

FinHack is a broad quantitative research framework. Its README describes a full workflow covering data collection, factor computation, factor mining, factor analysis, machine learning, strategy writing, backtesting, and live trading integration. The README also says the project resumed updates on 2025-03-11 and is undergoing major refactoring, so current code should be treated as research material rather than a drop-in stable dependency.

Its repo shape is framework-oriented:

- `collector`, `market`, and `factor` handle data and factor workflows.
- `trader` and project templates model backtest/live trading vendors.
- `core` builds a command-loader mechanism around `finhack <module> <action>`.
- `widgets/templates/empty_project` is used to scaffold user projects.
- The QMT-related code appears mostly as a template/simulation layer, not as a MiniQMT-specific read-only adapter.

FinHack's public license is GPL-3.0 with a commercial-license option. Do not copy code into this repository unless the project license decision is explicit and compatible.

## Current local MiniQMT baseline

The current repository already has a MiniQMT path:

- `stock_research/api/miniqmt.py` keeps xtquant optional, lazy-imported, and read-only. It masks account data and raises `LiveTradingDisabled` for order/cancel APIs.
- `stock_research/market/miniqmt_data.py` bridges through QMT's bundled Python to fetch market data, normalizes code format, persists CSV cache, and optionally writes 1d data into the existing kline repository.
- `stock_research/market/miniqmt_financial.py` builds strict point-in-time financial cache from MiniQMT financial rows.
- `apps/miniqmt.py` provides diagnostics, read-only account query, bar fetch, financial fetch, financial cache build, and price comparison.
- `apps/miniqmt_backtest.py` runs the existing portfolio replay with MiniQMT price frames.
- `stock_research/strategies/miniqmt_backtest.py` adds MiniQMT-like cost defaults, estimated slippage, close-proxy execution, and next-day signal effectiveness while still using the existing point-in-time portfolio backtest.

This is materially different from FinHack's QMT template. The local system is after-close research and replay first; it is not a live intraday event engine.

## Useful ideas to learn

1. Rule-chain execution simulation

FinHack's QMT template has a compact rule chain for delisting, ST, main-board filtering, suspension, limit-up/limit-down, board-lot rounding, volume participation, fees, slippage, and T+1. The local backtest already models some execution costs, but a future MiniQMT profile could benefit from an explicit `order_constraint` layer that records why a signal could not become an executable simulated order.

2. Vendor boundary

FinHack keeps a module/vendor/action shape and can load framework or user modules dynamically. The local repository should not adopt that broad loader wholesale, but the idea supports a narrower local boundary: keep `akshare`, `miniqmt`, and any future provider behind explicit adapters with capability probes and source metadata.

3. Factor metadata discipline

FinHack has factor list, factor analysis, and factor validity concepts. The local factor pipeline could use a lighter metadata manifest for factor provenance, point-in-time availability, freshness, and validation status, without taking on FinHack's MySQL/pickle-heavy factor store.

4. Project template thinking

FinHack's scaffolded project pattern is useful conceptually for separating framework defaults from user strategy code. Locally, that maps better to documented config/examples and typed strategy interfaces than to dynamic import of arbitrary strategy files.

## What not to import

- Do not enable live MiniQMT order APIs from FinHack's QMT template. The local invariant remains `live_trading_enabled=False`.
- Do not bypass the seven-step production pipeline: `formula33 -> sector_stats -> sector_watch -> factor_selection -> fundamental_update -> fundamental_selection -> daily_report`.
- Do not replace point-in-time data rules with FinHack's dynamic real-price or dynamic-adjustment assumptions without a dedicated audit.
- Do not copy GPL-3.0 code into this repository without an explicit licensing decision.
- Do not introduce FinHack's broad runtime loader as a dependency; it is too large for this repository's current stable surface.

## Suggested local follow-up

The safest next implementation is a local MiniQMT execution-constraint module, not a FinHack integration:

- Add a pure local `MiniQmtOrderConstraints` or similar dataclass/function.
- Inputs: date, code, side, target amount/value, latest daily bar, holding enable amount, account cash, configured fees/slippage.
- Outputs: adjusted amount, estimated value/cost, executable flag, and structured block reasons.
- Rules to model first: suspension/no trade, limit-up buy block, limit-down sell block, 100-share lot rounding, available shares/T+1, minimum commission, stamp duty, slippage, and max volume participation.
- Verification: focused unit tests plus one backtest fixture proving blocked signals are recorded without changing candidate generation.

## Three-role check

Main agent view: FinHack was studied as an external reference only. No production rule or code path was changed by this note.

Baida view: FinHack's strongest relevant idea is explainable execution constraints. It can improve "why not buy/sell" explanations, but it should remain subordinate to current value-line, Formula33, sector, right-side, and point-in-time rules.

Quant view: The main risks are license contamination, live-trading leakage, look-ahead through dynamic adjustment, and replacing stable pipeline boundaries with a broad plugin runtime. The low-risk path is to reimplement only the needed constraint concepts locally with tests.
