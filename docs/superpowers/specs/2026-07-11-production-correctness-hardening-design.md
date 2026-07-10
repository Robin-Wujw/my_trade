# Production Correctness Hardening Design

## Goal

Make the daily research pipeline fail closed when required market or sector data is incomplete, use a verified and coherent sector-data provider, guarantee that `--no-push` suppresses every PushPlus path, and make stock K-line persistence recover from transient DuckDB lock conflicts and CSV/database gaps.

The change protects correctness before publication. It does not attempt the larger `run_id` data-architecture migration.

## Scope

- Restore the documented Formula33 market-cap requirement in the production entry point.
- Standardize production sector analysis on the Tonghuashun industry taxonomy and verified non-Eastmoney endpoints.
- Reject stale or insufficient sector coverage before sector outputs can feed the daily report.
- Reconcile the current sector-board membership snapshot in DuckDB.
- Propagate top-level no-push behavior to fundamental coverage alerts.
- Retry transient DuckDB K-line writes and reconcile partial CSV/database persistence.
- Prevent the daily report from running after a critical sector-statistics failure.
- Add focused unit and integration tests for each failure mode.

## Non-Goals

- Do not connect to a broker or add order execution.
- Do not redesign strategy formulas or thresholds other than enforcing the existing RMB 10 billion (100 yi yuan) market-cap rule.
- Do not complete the full `RunContext`/`RunRepository` migration.
- Do not repair scheduled-task ACLs, package metadata, or dependency locking in this change.
- Do not silently manufacture sector results from sample data in the production entry point.
- Do not mix Tonghuashun and Eastmoney board histories inside one production run.

## Correctness Invariants

1. A formal Formula33 run applies `market_cap > 100` yi yuan. Missing market-cap values do not pass.
2. A sector board is fresh only when its latest business date is no more than seven calendar days before the observation date.
3. A production sector run uses one taxonomy and one price-history provider for every board.
4. Sector output requires at least 95% fresh-board coverage. Stale rows do not count toward coverage and do not enter the output frame.
5. `--no-push` suppresses final reports, failure alerts, factor messages, and fundamental coverage alerts.
6. A fetched stock K-line is not considered durably persisted until the CSV cache and DuckDB agree, or the step reports a persistence failure.
7. The daily report is not generated when Formula33, sector statistics, sector watch, or fundamental selection fails.

## Design

### 1. Production Formula33 Parameters

`stock_research.pipelines.daily.build_default_steps()` will use:

- `--market-cap-source auto`
- `--missing-mktcap-policy exclude`

`auto` keeps the existing source priority and error reporting. If no market-cap source can produce a usable map, Formula33 exits nonzero instead of switching to a technical-only result. Individual stocks with no market-cap value are excluded.

The Formula33 CLI may retain `none` and `pass` for explicit diagnostics, but the production orchestrator must never select them.

### 2. Sector Provider And Network Strategy

Production sector analysis will use Tonghuashun (`ths`) as one coherent provider family:

- board names and codes from the Tonghuashun industry list;
- current board summary from the Tonghuashun industry overview;
- daily board OHLCV from Tonghuashun industry-index history;
- board constituents from the paginated Tonghuashun industry detail pages;
- Shanghai Composite benchmark history from the verified Sina index endpoint.

This is a taxonomy change from the current Eastmoney hierarchy to the 90-board Tonghuashun industry universe. It is deliberate: on 2026-07-11 the production interpreter retrieved all 90/90 board histories through 2026-07-10 in 6.3 seconds, and paginated constituents for representative top boards without errors. The same host family supplies board names, codes, history, and constituents, avoiding name-to-code joins across incompatible taxonomies.

The adapter lives behind `stock_research.api` and applies explicit connect/read timeouts, bounded retries, and browser-compatible headers/cookies. Production does not depend on ambient Windows proxy settings. A short preflight fetches the board list and representative current history before the pipeline starts the full board loop.

Eastmoney remains an explicit diagnostic/manual provider, not an automatic per-board fallback. The current machine reaches Eastmoney DNS, TCP, and TLS but receives HTTP resets through both direct and local-proxy routes. Code may support an explicit `EASTMONEY_PROXY` for manual recovery, but production never mixes Eastmoney rows with Tonghuashun rows after one board fails.

The Eastmoney diagnostic adapter will also paginate the board list according to the API's reported total. The server caps a response at 100 rows even when `pz=5000`; treating that response as complete caused the observed 100-current/396-old membership split.

### 3. Sector Membership Snapshot

The board repository will expose snapshot replacement semantics for active industry-board membership. A successful network refresh writes the current board set transactionally and removes obsolete active rows regardless of their previous CSV or network source label. Historical K-lines are retained; only active membership is reconciled.

Board-list reads use the latest coherent snapshot rather than merging a partial 100-row network response with 396 retained CSV-import rows.

### 4. Sector Freshness And Coverage Gate

Both sector pipelines will use a shared pure validation helper with these inputs:

- expected board names;
- per-board history frames;
- observation date;
- `max_stale_days=7`;
- `min_fresh_coverage=0.95`.

The helper returns fresh histories and diagnostics containing expected, fresh, stale, missing, and coverage counts. It never rewrites dates and never treats `updated_at` as a business-data date.

Repository reads are provider-aware. After the switch, active calculations query only `ths` rows; old Eastmoney histories can remain for audit but cannot satisfy Tonghuashun coverage.

When a Tonghuashun request fails, an old CSV/DuckDB frame may be retained for diagnostics, but it is marked stale and excluded from calculations. The loader no longer returns stale history as ordinary success and does not switch that individual board to another taxonomy.

The pipeline aborts once the number of stale or missing boards makes 95% coverage mathematically impossible, avoiding hours of retries that cannot lead to publication. Production sector retries are reduced from five to two per request; callers can still override the value for manual recovery runs.

On gate failure, the step logs a structured coverage event and exits nonzero without writing a new sector statistics/watch export.

### 5. Limit-Up And Benchmark Alignment

The current limit-up pool remains a separate daily input because its endpoint is reachable on this machine. Industry strings from that pool are not joined to Tonghuashun boards by raw text: suffixes such as `II/III`, known truncations, and unique-prefix aliases are normalized first. For the final top boards, limit-up membership is verified by intersecting limit-up stock codes with the fetched Tonghuashun constituent sets.

Limit-up responses are cached by trade date. A missing required date without a cache is a gate failure rather than an all-zero breadth reading.

The Shanghai Composite benchmark switches from `stock_zh_index_daily_em` to the verified Sina `stock_zh_index_daily(symbol="sh000001")` path. An empty benchmark is a gate failure because resilience/attack scores otherwise become silently incomplete.

### 6. Push Suppression

The daily step builder adds `--alert` to `fundamental_update` only when top-level `no_push` is false. Existing factor and report no-push behavior remains unchanged, and the pipeline-level failure alert remains guarded by `not no_push`.

Tests will execute the generated closures with spies and prove that no PushPlus call is reachable when `no_push=True`.

### 7. K-Line Persistence Recovery

`KlineRepository` will retry the whole connect/transaction operation for transient DuckDB file-lock errors after it has acquired the existing process lock. Retries are bounded and use short backoff. Non-lock database errors are raised immediately.

`load_kline_with_cache()` will compare CSV and DuckDB trade-date sets. When CSV contains dates absent from DuckDB, it upserts those rows even when the database is not empty. This repairs the observed one-day partial gap instead of only backfilling an entirely empty database.

Fresh API data is written to the CSV cache before the database attempt so it remains recoverable. If bounded database retries still fail, the stock result records a persistence failure and the Formula33 step exits nonzero after workers finish; it must not report a fully successful durable run.

### 8. Daily Report Gate

`run_daily_pipeline()` retains independent diagnostic execution where useful, but `daily_report` requires successful statuses for:

- `formula33`;
- `sector_stats`;
- `sector_watch`;
- `fundamental_selection`.

A failed requirement adds `daily_report` to skipped steps and keeps the process exit code nonzero through the original failed step.

## Error Reporting

Console and `ops.pipeline_events` messages will distinguish:

- source request failure;
- stale cache available but rejected;
- board coverage gate failure;
- transient DuckDB lock retry;
- exhausted persistence retries;
- CSV-to-DuckDB reconciliation.

Messages include board/code, latest business date, observation date, coverage counts, retry attempt, and the final exception where applicable. Credentials are never included.

## Test Strategy

Tests are written before implementation and cover:

1. Production Formula33 arguments use `auto/exclude` and reject `none/pass`.
2. Top-level `--no-push` omits the fundamental alert path.
3. Tonghuashun board-list, history, summary, and constituent adapters normalize the verified response shapes.
4. Production never combines Tonghuashun and Eastmoney histories in one result.
5. Sector validation accepts 86 fresh boards out of 90 and rejects 85.
6. A 2022 cache returned after source failure is classified stale and excluded.
7. Sector loading aborts when 95% coverage becomes impossible and writes no export.
8. Board snapshot replacement removes obsolete active members without deleting history.
9. Limit-up aliases and constituent-code intersections map to Tonghuashun boards.
10. A missing benchmark or required limit-up date fails the gate.
11. A transient DuckDB lock failure is retried and succeeds.
12. A partial DuckDB history is repaired from the newer CSV cache.
13. Exhausted persistence retries make Formula33 fail rather than silently succeed.
14. Daily report is skipped when sector statistics fails.

The final verification sequence is:

```powershell
& 'D:\ActionsRunner\my-trade\python\python.exe' -m compileall -q apps stock_research tests
& 'D:\ActionsRunner\my-trade\python\python.exe' -m pytest -q
& 'D:\ActionsRunner\my-trade\python\python.exe' -m stock_research.regression.output_baseline verify tests/regression/legacy-output-v1.json
& 'D:\ActionsRunner\my-trade\python\python.exe' -m apps.daily_pipeline --dry-run --no-push
.\scripts\run_daily_analysis.ps1 --no-push
```

The production run passes only if all seven steps complete with current data. If external sector sources remain unavailable, the accepted result is a bounded, explicit nonzero failure with no stale sector or daily-report output.

## Rollout

1. Land focused tests and implementation without changing strategy output formats.
2. Run offline and regression verification.
3. Run the Tonghuashun preflight and require a current board list, current benchmark, and at least one current board history.
4. Run one production no-push execution with PushPlus credentials disabled for the process.
5. Require at least 95% current Tonghuashun board coverage and verify top-board constituents.
6. Compare Formula33 CSV/DuckDB dates and sector coverage diagnostics.
7. Re-enable scheduled execution only after a complete gated run succeeds.
