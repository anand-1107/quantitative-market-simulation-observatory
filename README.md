# Quantitative Market Simulation & Prediction Observatory

A research-oriented Python package for stochastic market simulation and historical market analysis. Historical summaries describe past observations; they do not predict future prices.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Architecture

- `quantum_market_lab.gbm`: NumPy Geometric Brownian Motion simulation.
- `quantum_market_lab.api`: FastAPI health, simulation, calibration, evaluation, market-data, event/news, forensics, and experiment endpoints.
- `quantum_market_lab.data.market_data`: yfinance downloads, normalization, and quality reports.
- `quantum_market_lab.data.events` and `news`: historical event thresholds, event-window diagnostics, optional Finnhub news retrieval, matching, and caches.
- `quantum_market_lab.data.features`: returns, volatility, drawdown, and annualization.
- `quantum_market_lab.data.storage`: local Parquet datasets and JSON metadata.
- `quantum_market_lab.data.experiments`: transactional SQLite experiment metadata and compact results.
- `quantum_market_lab.data.analytics`: DuckDB summary queries over stored Parquet data.
- `quantum_market_lab.models`: OU, Merton jump-diffusion, GARCH(1,1), diagnostics, and model registry.

## Run

```bash
pytest
python -m uvicorn quantum_market_lab.api.app:app --reload
```

Open interactive API documentation at <http://127.0.0.1:8000/docs>. Health check: <http://127.0.0.1:8000/health>.

The GBM experiment can be run with `python -m quantum_market_lab.main` when the package is installed editable.

### Environment configuration

All variables are optional unless you use their associated feature. Defaults are relative to the backend working directory unless noted.

| Variable | Used by | Default / purpose |
| --- | --- | --- |
| `FINNHUB_API_KEY` | Backend only | Unset. Enables live historical company-news retrieval. Never put this key in frontend configuration or commit it. |
| `QML_EXPERIMENT_DB` | Backend | `data/experiments.sqlite3`; SQLite experiment and path-forensics metadata. |
| `MARKET_EVENT_NEWS_CACHE` | Backend | `data/news_cache.sqlite3`; optional article response cache. |
| `MARKET_EVENT_ANALYSIS_CACHE` | Backend | `data/event_analysis_cache.sqlite3`; optional event-analysis response cache. |
| `VITE_API_BASE_URL` | Vite frontend | `http://127.0.0.1:8000`; copy `frontend/.env.example` to `frontend/.env.local` to override. |

Keep credentials in the backend process environment or a local, untracked secrets manager. The API returns provider configuration status, never the Finnhub key.

## Historical market data

These endpoints access Yahoo Finance and may require internet connectivity. The provider can change, rate-limit, or fail; its availability and historical data quality are not guaranteed. No data is fabricated when downloads fail.

- `GET /market-data/validate/AAPL?start=2024-01-01&end=2025-01-01&interval=1d`
- `GET /market-data/summary/SPY?start=2024-01-01&end=2025-01-01`
- `POST /market-data/download` with `{"ticker":"AAPL","start":"2024-01-01","end":"2025-01-01","interval":"1d"}`

Downloads saved through the API are stored under `data/market/{TICKER}/daily.parquet`, with `metadata.json` alongside. Existing datasets are not overwritten. The storage root can be configured when using `MarketDataStore` directly.

Normalization uses lowercase OHLCV fields and optional `adj_close`, UTC dates converted to timezone-naive calendar dates, ascending order, and first-observation-wins for duplicate dates. OHLC prices must be finite and positive and OHLC relationships must be coherent. Zero volume is valid. Log returns prefer adjusted close when present. Daily volatility is the sample standard deviation of daily log returns; annualized volatility multiplies by the square root of the configured trading-period count (252 by default). Annualized mean log return is a continuously compounded rate and differs from annualized arithmetic return.

Yahoo Finance data accessed through yfinance is suitable for exploratory analysis, not guaranteed as an authoritative or complete record. Corporate actions, missing observations, symbol changes, provider adjustments, and outages can affect results. Historical analysis and calibrated stochastic simulations are research tools; neither establishes future performance.

## Stochastic model laboratory

All simulation routines use `numpy.random.default_rng(seed)`. Fixed seeds give reproducible output for a given NumPy implementation. Price-level models return `(times, paths)` where time has `steps + 1` entries and the path matrix is `(paths, steps + 1)`. GARCH is a discrete-period return model and returns `(returns, conditional_variances)` with shape `(paths, steps)`.

### Models and conventions

- **GBM** (`simulate_gbm`): `dS/S = drift*dt + volatility*dW`. Drift and volatility are annualized; the exact lognormal transition uses `drift - volatility²/2`. Prices stay positive. Constant parameters, Gaussian diffusion, and continuous trading are simplifying assumptions.
- **Ornstein–Uhlenbeck** (`simulate_ou`): `dX = theta*(mean_level-X)*dt + volatility*dW`. Uses the exact conditional Gaussian step, with theta in inverse years and volatility in level units per square-root year. For theta > 0, stationary mean is `mean_level`, stationary variance is `volatility²/(2*theta)`. It is suited to spreads or transformed variables and is not a complete stock-price model.
- **Merton jump-diffusion** (`simulate_jump_diffusion`): GBM diffusion plus normally distributed log jumps arriving at constant Poisson intensity. `drift` denotes expected proportional asset growth; the jump compensator is included so theoretical `E[S_T] = S_0*exp(drift*T)`. The specified diffusion volatility excludes jump contribution, so total return variability depends on jump intensity and jump sizes too. Constant intensity and independent jumps are model limitations.
- **GARCH(1,1)** (`simulate_garch`): `h[t] = omega + alpha*epsilon[t-1]² + beta*h[t-1]`, `epsilon[t] = sqrt(h[t])*z[t]`. Parameters are per observation period. `omega > 0`, `alpha,beta >= 0` ensure positive conditional variance; covariance stationarity requires the stronger `alpha+beta < 1`. The default initial variance is the unconditional variance for stationary parameters, and `omega` for a valid nonstationary parameterization. Returns use the selected `log` or `simple` convention; `garch_returns_to_prices` compounds accordingly.

The model registry is accessible in Python through `quantum_market_lab.models.list_models()` and over `GET /models`. The summary routes (`POST /simulate-gbm`, `/simulate-ou`, `/simulate-jump-diffusion`, `/simulate-garch`) return compact statistics. For bounded, actual trajectory data, corresponding visualization routes are available at `POST /simulate-gbm/paths`, `/simulate-ou/paths`, `/simulate-jump-diffusion/paths`, and `/simulate-garch/paths`. These return time coordinates, model parameters, trajectories, per-coordinate P05/P50/P95 bands, and empirical terminal summaries. GARCH also returns conditional variances; its Y-axis quantity is per-period return, not price. Trajectory responses are limited to 128 paths and 512 steps. The summary API separately caps total path values at 5,000,000 per request, which keeps the original GBM default request within the limit.

### Historical model calibration and GBM evaluation

`POST /calibration/gbm` accepts `{ "ticker": "AAPL", "start": "2020-01-01", "end": "2024-01-01", "interval": "1d" }` and reuses the validated market-data loader. Dates use a half-open `[start, end)` window; observations outside that window or out of chronological order are rejected. Supported calibration intervals are `1d` (252 periods/year), `5d` and `1wk` (52), `1mo` (12), and `3mo` (4); intraday annualization is deliberately unsupported. Prices use `adj_close` when present and otherwise `close`. For period log returns `r`, the estimator uses sample standard deviation (`ddof=1`): `sigma = sd(r) * sqrt(N)` and annual GBM price drift `mu = mean(r) * N + 0.5 * sigma^2`, where `N` is the interval annualization factor. Drift is expected proportional price growth per year in the simulator's `dS/S = mu dt + sigma dW` convention. The estimator assumes equally spaced observations, constant parameters, and independent Gaussian diffusion increments; annualization factors are conventional approximations. Estimates with fewer than 30 returns include a warning; historical parameters are not forecasts.

Additional typed endpoints are `POST /calibration/ou`, `/calibration/jump-diffusion`, and `/calibration/garch`:

- **OU:** conditional least squares fits the exact discrete AR(1) transition and converts `phi` to continuous time using `theta=-log(phi)/dt`; innovation variance is converted to the simulator's diffusion sigma. The user selects log return, log price, or raw price. Price levels require explicit confirmation and receive a suitability warning. The fitted coefficient must satisfy `0 < phi < 1`; inputs with too few observations or no identified mean reversion are rejected.
- **Merton:** deterministic multi-start maximum likelihood fits the exact finite-sampled compound-Poisson/normal mixture used by the simulator. Drift is expected proportional growth after the jump compensator; intensity is annual, and jump sizes are normal in log space. A three-standard-deviation tail rule is used only for optimizer initialization, not to declare observations as jumps. Sparse expected jumps, boundary estimates, and optimizer failures are reported as warnings/status.
- **GARCH(1,1):** Gaussian conditional maximum likelihood fits log returns at the selected observation frequency. A constrained parameterization enforces `omega>0`, `alpha,beta>=0`, and `alpha+beta<0.999`. Conditional volatility estimates are returned with dates. The fitted unconditional variance initializes the existing simulator consistently; all parameters remain per observation interval.

All three services use the validated historical loader and provide model-specific provenance, assumptions, sample counts, and warnings. Simulations accept only matching model-specific calibration metadata and check that calibrated parameter values match the submitted controls. These estimators remain descriptive historical fits; jump-mixture estimates in particular can be weakly identified, and extreme observations are not proof of jumps.

The response can be passed as the optional `calibration` metadata field to `POST /simulate-gbm/paths` alongside the same `drift` and `volatility` values. The path response retains ticker, estimation dates, annualization inputs, sample size, method, and calibration timestamp. Without this field the existing manual-parameter workflow is unchanged.

`POST /evaluation/gbm` accepts ticker, separate estimation and evaluation date ranges, the same interval, a path count, and a seed. Evaluation must begin after the estimation end; the API also verifies the downloaded observations do not overlap. It calibrates only on the first range, then simulates the held-out horizon. Returned diagnostics compare sample annualized volatility, period log-return quantiles, observed first-to-last simple return with simulated terminal-return quantiles, and historical/simulated maximum drawdown. The historical evaluation window contains one terminal return, so its terminal-return quantiles are explicitly unavailable and the observed value is reported as a scalar. These are descriptive out-of-sample comparisons, not claims of predictive accuracy or profitability. Calibration and evaluation outputs are not persisted.

Example calls:

```bash
curl -X POST http://127.0.0.1:8000/simulate-ou -H 'Content-Type: application/json' \
  -d '{"initial_value":0,"mean_level":1,"mean_reversion_speed":2,"volatility":0.5,"years":1,"steps":252,"paths":1000,"seed":42}'
curl -X POST http://127.0.0.1:8000/simulate-jump-diffusion -H 'Content-Type: application/json' \
  -d '{"initial_price":100,"drift":0.08,"volatility":0.2,"jump_intensity":0.5,"mean_log_jump":-0.1,"jump_volatility":0.2,"years":1,"steps":252,"paths":1000,"seed":42}'
curl -X POST http://127.0.0.1:8000/simulate-garch -H 'Content-Type: application/json' \
  -d '{"omega":0.000001,"alpha":0.05,"beta":0.9,"steps":252,"paths":1000,"seed":42,"return_convention":"log","initial_price":100}'
```

The parameterized simulation routes accept manual inputs. Historical calibration is available only through the explicit calibration routes documented above; simulated distributions remain model-based scenarios and are not automatically market forecasts.

## React research dashboard

The Vite frontend lives in `frontend/` and uses React, TypeScript, Recharts, Lucide React, Three.js, React Three Fiber, and Drei. The sidebar pages are Overview, Market Data, Model Laboratory, Quant Formula Lab, Simulation Universe, Risk Analytics, Experiments, Market Event Intelligence, and Model Tournament. The Model Comparison Observatory is a 3D research view inside the tournament. The UI reads the shared model registry and health status from the API, calls existing market, simulation, calibration, event/news, evaluation, and experiment endpoints, and displays returned results with explicit loading, empty, error, or unavailable states. The Experiments page lists saved runs, opens model-specific metrics and provenance, compares compatible runs, and can restore a configuration in the model laboratory.

The historical candlestick chart uses `GET /market-data/ohlcv/{ticker}` and displays actual OHLCV bars from the configured market-data loader. The endpoint allows at most 5,000 output bars. If the source range is longer, it aggregates contiguous observations while preserving the first open, maximum high, minimum low, last close, and summed volume. The frontend requests up to 1,500 bars and labels aggregated output. Crosshair details, drag panning, wheel zoom, price-axis scaling, volume bars, and optional SMA/EMA overlays operate on those returned observations. Timeframes use intervals accepted by the current Yahoo Finance loader; provider retention limits can restrict shorter intervals. The existing `GET /market-data/prices/{ticker}` contract remains available for consumers of adjusted/close price series.

The instrument watchlist contains symbols only; it does not show fabricated quotes. Selecting a ticker shares it with the model laboratory. Historical closes, period statistics, and simulation trajectories remain labeled by source and are fetched or computed through their respective backend workflows.


The simulation cockpit uses the bounded Python path routes above. They return quantile bands and a `terminal_risk` object for applicable price and return paths; OU state levels correctly report risk as unavailable. Terminal scenario return is a simple return: `R = S_T/S_0 − 1` for price paths, `exp(sum(log_returns)) − 1` for log-return paths, and `product(1 + simple_returns) − 1` for simple-return paths. Loss is `L = −R`; VaR is the confidence quantile of empirical losses using linear interpolation. ES is the arithmetic mean of all observed losses `L_i >= VaR`, including ties, so the tail count can exceed the nominal tail fraction. Both are fractions of initial wealth over the full simulation horizon, not per-period or annualized values. Responses include confidence, horizon, scenario count, quantile convention, and per-path total return and maximum drawdown. Small simulation samples receive a warning.

The dedicated **Simulation Universe** page uses these same Python-generated trajectory endpoints and existing model calibration endpoints. At each returned time coordinate, it estimates a cross-sectional Gaussian KDE across the actual paths and evaluates it on one common value grid. The bandwidth follows Scott's normal-reference rule (`1.06 × sample SD × n^−1/5`), with a half-grid-spacing floor to keep constant or near-constant samples renderable; the 3D Z axis is density per price, return, or state unit. Density estimation requires at least two paths, validates finite and equally sized paths, and uses a bounded 32–128 point grid. Playback reveals every returned path at one shared time index in both the 3D surface and chart, while progressively revealing the KDE surface, selected-path highlight, and the backend's ensemble median and 5–95% band. The immutable response is reused for play, pause, reset, replay, and scrubbing; no paths are regenerated during animation. Rendering honors the path API limits of 128 paths and 512 steps. Charts show simulated price, return, or state values, not strategy P&L. OU levels and GARCH per-period returns retain their distinct units and labels. Path metrics and effective parameters come from the simulation response, and calibration is explicitly applied to a matching model before a new run.

### Path forensics and scenario matching

Trajectory endpoints accept the additive `retain_forensics: true` option (the Simulation Universe sends it automatically). Opted-in runs receive a UUID `forensics_run_id` and store the original trajectories, time grid, effective configuration, calibration provenance, and per-path innovations in the same SQLite database selected by `QML_EXPERIMENT_DB`. The newest 25 forensic runs are retained; older forensic records are pruned when a new record is saved. Runs without the option preserve the existing no-storage behavior. `GET /path-forensics/{run_id}/paths/{path_id}` is one-based and returns the selected original path, model version, intermediate transition components, innovations, terminal diagnostics, reconstructed values, maximum absolute reconstruction error, and verification status.

Reconstruction uses the existing model conventions: GBM replays standard-normal diffusion increments; OU replays exact conditional-mean and diffusion steps; Merton replays Poisson counts, the simulator's aggregate jump normal/log-jump sum, and diffusion normals with its jump compensator; GARCH replays standard-normal shocks through the stored initial variance and `h[t+1] = omega + alpha*epsilon[t]^2 + beta*h[t]` recursion. Explicit innovations provide path-level replay evidence; the seed is retained as provenance but is not used as the reconstruction mechanism. The Merton engine samples one aggregate normal per time step rather than separate per-jump normals, so forensics reports its actual aggregate jump component.

`POST /path-forensics/scenario-matches` performs a bounded reference-path search with 1–32 parameter candidates. Candidates are validated against the matching model's API schema and evaluated with the reference path's stored innovations as common random numbers. The score is normalized path RMSE (scale is `max(reference SD, 1% of reference scale, 1e-8)`) plus optional weighted absolute terminal simple-return and maximum-drawdown differences where those quantities are defined. OU state paths use path RMSE only. This is a deterministic in-sample counterfactual match, not a forecast or unique causal explanation; the Simulation Universe offers a small bounded set of parameter perturbations around the selected run.

`GET /market-data/summary/{ticker}` returns historical one-period VaR/ES separately. It converts close-to-close log returns to simple returns before calculating loss `L = −R`; the interval is one provider observation period, and the empirical sample size and tie convention are reported. Historical maximum drawdown is based on the adjusted-close series when complete and valid, otherwise close, and uses its running peak. Drawdown from a simple-return series includes initial wealth of one. Portfolio-level risk is not implemented. Scenario analysis reruns the selected stochastic simulator using explicit hypothetical drift, volatility, jump, mean, or state-level parameter changes; it is not a forecast. The Three.js cockpit, calibration, and chronological GBM evaluation remain available.

## Persistent experiments

`POST /experiments` validates a model-specific trajectory configuration, runs the existing simulator, and stores a compact completed record. It returns both the record and generated trajectories for immediate cockpit rendering; trajectories are not written to storage. `GET /experiments` lists records; `GET /experiments/{id}` and `/summary` retrieve configuration/provenance and metrics; `GET /experiments/compare?experiment_ids=...` returns only common numeric metrics when quantity types, units, and horizons match; `DELETE /experiments/{id}` removes a completed or failed record. Failed executions are retained with a failed status and error, never as completed. Input validation failures do not create a record. Synchronous runs transition through pending/running/completed or failed states.

Experiment metadata is stored in a separate SQLite database at `data/experiments.sqlite3` by default, configurable through `QML_EXPERIMENT_DB`. SQLite transactions and WAL keep this write workload isolated from the Parquet market store and DuckDB analytical reads. Metadata uses schema version 1 and contains UTC timestamps, the full effective typed parameter set, the supplied parameter names, derived/default parameters, calibration provenance, ticker when supplied, horizon, path count, seed, and Python/NumPy/application versions. Trajectory matrices are not persisted; compact terminal quantiles, conditional variance summaries, and risk measures are. Re-run the saved configuration in the model cockpit to regenerate trajectories. A fixed seed is reproducible under the same model implementation, NumPy version, and runtime assumptions; unseeded experiments are explicitly marked as non-reproducible. GARCH's derived initial variance is materialized in the stored effective configuration.

The historical summary's period volatility is the sample standard deviation (`ddof=1`) of valid period log returns; annualized volatility is that same full-sample statistic multiplied by `sqrt(periods_per_year)` for the selected interval. This avoids comparing a full-sample period volatility to a different rolling-window annualized statistic.

## Model Tournament and Comparison Observatory

`POST /model-tournaments` takes a ticker, one or more of `gbm`, `ou`, `jump_diffusion`, and `garch`, two half-open date windows, an interval, a bounded path count, a seed, and an optional save flag. The backend downloads the training and test windows separately, requires a chronological gap between them, and fits each model using training data only. It then scores conditional one-step predictions across the test window in chronological order; after each forecast is scored, the newly observed return may update the next state or variance. The common target is the within-test close-to-close natural-log return, preferring adjusted close when available. The response includes each model's status, fitted values, assumptions, warnings, unavailable reason, metrics, and compact predictive-density and risk surfaces. A fit failure remains visible as `unavailable`; it is not replaced by a score.

Directly comparable diagnostics include mean negative predictive log score (nats per observed return), PIT Kolmogorov–Smirnov statistic, QLIKE volatility score, and the one-period 95% VaR breach rate. PIT KS is descriptive and is not adjusted for rolling dependence or used as a stand-alone significance test. VaR breach rates need a sufficiently long holdout. Historical test VaR/ES are ex-post sample summaries, while forecast VaR/ES are calculated from each model's predictive distribution before the realized observation. Terminal-return and maximum-drawdown distributions come from equal-horizon seeded simulation and are model-based scenarios, not point forecasts. They are model-specific outcomes and do not establish that one model is universally best.

OU is currently configured in the tournament with **log return per observation interval** as its mean-reverting state. This is a valid, explicit state definition, not an OU stock-price process. Exact sampled OU transitions require the training AR(1) coefficient to be in `(0, 1)`. When ordinary asset returns have no positive, stable serial dependence, OU calibration can therefore fail; that is a data/model compatibility diagnostic, not evidence that calibration silently substituted another series. Use a sufficiently long sample of a series with a defensible mean-reverting interpretation. The current tournament UI does not offer an alternate OU state-series selector.

The Model Comparison Observatory reuses the tournament's already-computed rolling-origin conditional density and risk outputs. Its Distribution Evolution, Risk Landscape, and Model Disagreement surfaces are derived from those saved forecasts; Path Comparison calls the existing seeded trajectory endpoints only when requested. A single playback clock progressively reveals visible prefixes across modes. Playback does not refit models, download data, or generate new paths on each frame. Reset and replay reuse the stored response. The surface grid is a compact display sample (up to 64 forecast origins by 49 return-grid points), not the full test history. Compatible density disagreement requires identical out-of-sample dates, return grid, and units.

API routes: `POST /model-tournaments` runs the workflow; `GET /model-tournaments` lists saved model rows grouped into tournament runs. Saving uses the existing SQLite experiment database and stores model-specific status, failure reason when unavailable, metrics, parameters, assumptions, calibration provenance, dates, seed, and compact research surfaces. Trajectories themselves are not persisted by the experiment store; explicit path forensics is a separate, bounded opt-in store.

Start the backend from the project root:

```bash
.venv/bin/python -m uvicorn quantum_market_lab.api.app:app --reload
```

In another terminal, start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The frontend's default API URL is `http://127.0.0.1:8000`. To use a different backend, copy `frontend/.env.example` to `frontend/.env.local` and set `VITE_API_BASE_URL` (for example `http://127.0.0.1:8000`). Restart Vite after changing the value. FastAPI explicitly allows the localhost Vite origins on port 5173.

Run the frontend checks from `frontend/` with `npm run typecheck` and `npm run build`. Run backend tests from the repository root with `.venv/bin/pytest -q`.

### Portfolio demonstration flow

1. Open Overview and confirm the API model registry is available.
2. Load a historical instrument and date range in Market Data; inspect the actual OHLCV candles and data-quality summary.
3. In Model Laboratory, run a seeded simulation and inspect the model's quantity, parameter source, and returned paths. Apply a historical calibration only when its state definition and interval suit the model.
4. In Simulation Universe, run a bounded seeded simulation and use synchronized play, pause, scrub, reset, and replay to inspect the ensemble paths and KDE surface.
5. In Model Tournament, choose separate training and later test windows, run a small subset first, then inspect per-model fit status, assumptions, warnings, predictive scores, and the synchronized 3D observatory. Save the run and reopen it from Saved Tournament Runs.
6. Use Risk Analytics for explicitly requested historical risk and Market Event Intelligence for observed event evidence. Treat event news as optional: live retrieval requires a Finnhub key and account access.

For a concise walkthrough, emphasize train/test separation, one-step rolling-origin scoring, fixed-seed simulation, and the difference between conditional distributions and point forecasts. A model can legitimately be unavailable for a selected series or sample; keep that diagnostic visible rather than omitting the model from the discussion.

## Market Event Intelligence

The sidebar's Market Event Intelligence page uses the existing Yahoo Finance OHLCV loader through `POST /market-events/analyze`. Candidate triggers are configurable close-to-close absolute log returns, volume versus a prior-only rolling median, session open gaps, rolling realized-volatility spikes versus an earlier median, and sustained cumulative moves. Detection and event-window metrics use observed bars; invalid bars and large timestamp gaps are reported and returns crossing those gaps are suppressed. Display bars are capped at 2,500 observations while calculations use all valid source bars.

`POST /market-events/analyze` returns candidate evidence, descriptive pre/post windows, optional benchmark-relative returns, and in-sample GBM, Merton, and GARCH diagnostics where existing calibrators support the interval and sample. Event windows compare pre-event closes with the event close and subsequent observations. Returns are simple close-to-close window returns; realized volatility is sample standard deviation of per-observation log returns; maximum drawdown uses each window's running close peak. These comparisons do not establish causation. Model fits use the full selected period and may include the event itself, so their rarity measures are in-sample diagnostics, not pre-event forecasts.

Historical news is an optional Finnhub company-news integration. Configure `FINNHUB_API_KEY` in the **backend process environment only**; never put it in frontend environment variables or source code. `GET /market-events/news/provider` reports whether the server has this setting. `POST /market-events/news` retrieves historical articles, keeps provider headline, publisher, URL, publication timestamp, retrieval timestamp, ticker relevance, and excerpt, and ranks matches using visible temporal, instrument, category, and publisher-corroboration signals. Finnhub documents the company-news endpoint for North American companies; historical depth and access depend on the account plan. Without the credential, the API explicitly reports that no search was performed and returns no fabricated articles. Date ranges are half-open in this application; requests are converted to Finnhub's inclusive end date.

Exact news queries are cached in `data/news_cache.sqlite3` for one hour, configurable with `MARKET_EVENT_NEWS_CACHE`. Identical analysis requests are cached in `data/event_analysis_cache.sqlite3` for 15 minutes, configurable with `MARKET_EVENT_ANALYSIS_CACHE`. Publication-to-session classification is limited: session-date bars cannot establish pre-market, intraday, or after-market timing, and intraday comparisons are labeled relative to the observed bar because this application has no exchange-calendar integration.
