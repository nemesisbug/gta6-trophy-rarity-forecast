# Tide & Turn

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nemesisbug/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nemesisbug/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for how many GTA VI players will have earned each trophy by 2026-11-26 00:00 UTC, built for the make.fun competition gta6-trophy-rarity-forecast-01. It reads the Exophase earn rates the house recorded up to the deadline and forecasts each trophy's rate as five quantiles (5%, 25%, 50%, 75%, 95%) on the logit scale.

## The idea: tide and turn

A trophy's earn rate is a ratio, `rate = earners / players tracked`, and both halves move for different reasons. The **tide** is the player base: Exophase refreshes bring in a flood of new, less-progressed players, which dilutes every rate at once. The **turn** is the earners: as the existing cohort ages into the play-hours each trophy needs, earners grow — slowly at first, then fast, then saturating. A forecaster that extrapolates the rate directly confuses the two and misses both. Tide & Turn models them separately and divides at the target.

- **Earners — structural cohort model.** Players join as `N(t)` grows; each player has an engagement `e ~ LogNormal(m, s)` hours per day, and a trophy needs `H` hours of play. Earners are the cohort integral over join times. `H` is fitted per trophy on the recent increments with a Poisson deviance (flat trophies stay flat); the engagement level and dilution are picked by weighted SSE over all trophies. A Hill (log-logistic) curve in time is blended in as a second view — it allows the accelerating phase as cohorts age into a trophy, where a power law or saturating curve would under-predict three- to five-fold.
- **Players tracked — three-way fit.** The `N(t)` curve is fitted as a power law, a saturating exponential, and a Hill curve; the median is taken, capped at the observed growth rate extended.
- **Flat history — refresh ensemble.** When no rate has moved yet (launch night), three structural refresh models agree on the dilution: a power-law decay in `N` calibrated to the live page, class factors from the trophy descriptions, and a sparse-cohort split of the ~103 new trophies the first refresh brought. Their geometric mean is the median.
- **Quantile spread — honest, not narrow.** The spread combines the fit residual, the disagreement between the curve views, a horizon term, and the jumpiness between successive Exophase reads, floored at 0.40 logits. Overconfident narrow intervals are punished by pinball loss, so the spread is sized to the model's actual disagreement.

Rare trophies get special handling: when a trophy's earners are a handful of pioneers with no recent growth, its structural forecast is capped near persistence instead of extrapolating a phantom crowd.

## Validation

Rolling-origin backtesting on the practice cases and on a cohort simulator calibrated to the published Exophase anchors (638 players at launch, 3,868 at the first refresh, ~9,200 by midday):

| Cut | Tide & Turn | Persistence | Trend |
| --- | --- | --- | --- |
| Practice p00/p01 (synthetic answers A/B/C) | **0.157–0.180** | 1.39–1.67 | 1.39–1.67 |
| Simulated finals, 18 configs (mean / worst) | **0.333 / 0.450** | 0.539 / 0.738 | 0.654 / 0.968 |

Pinball loss on the logit scale, lower is better. Coverage of the 90% interval on the simulated finals is 0.87–1.00. The practice answers are synthetic (the true ones are only published as aggregates), so the practice score is a bracket, not a promise — but the model beats both house baselines by a wide margin on every cut tested.

## Run it

```sh
python3 main.py
```

It reads every `*.in` case from `$MAKEFUN_INPUT_DIR` (default `/input/`) and writes `/output/<id>.out` (or `$MAKEFUN_OUTPUT_DIR`). Pure standard library, deterministic, no network, no clock — a practice case runs in about 0.06 s.

## Layout

- `main.py` — the forecaster (the entry point).
- `dev/` — the evaluation harness, the cohort simulator, and the tuning scripts behind the numbers above. `dev/evalharness.py` scores a forecast against the synthetic practice answers; `dev/simulator.py` + `dev/eval_final.py` run the simulated-final backtests.
