# Architecture: Dengue Outbreak Early Warning

A weather-driven classifier that gives each ward a **Low / Medium / High / Critical** risk tier for the **next 28 days**, with a recommended action. The model is trained offline on 10 years of weekly ward data and serves predictions in a Streamlit app using live weather from Open-Meteo.

## 1. Data flow

```text
OFFLINE TRAINING (Colab / src/train.py)

+--------------------------------------------------------+
| Raw dataset: 31,320 weekly rows, 60 wards, 2016-2025   |
+--------------------------------------------------------+
                             |
                             v
+--------------------------------------------------------+
| Preprocess: drop max/min temp, build per-ward lags     |
+--------------------------------------------------------+
                             |
                             v
+--------------------------------------------------------+
| Target: dengue cases in the next 4 weeks,              |
| split into Low / Medium / High / Critical              |
+--------------------------------------------------------+
                             |
                             v
+--------------------------------------------------------+
| Chronological split:                                   |
| train 2016-2023, test 2024-2025                        |
+--------------------------------------------------------+
                             |
                             v
+--------------------------------------------------------+
| HistGradientBoosting (balanced class weights)          |
+--------------------------------------------------------+
                             |
                             v
+--------------------------------------------------------+
| Saved model: models/dengue_model.joblib                |
+--------------------------------------------------------+


ONLINE PREDICTION (Streamlit app)

+--------------------------------------------------------+
| User picks city + ward (date = today)                  |
+--------------------------------------------------------+
                             |
                             v
+--------------------------------------------------------+
| data/ward_static.csv: lat/lon + 3 fixed ward features  |
+--------------------------------------------------------+
                             |
                             v
+--------------------------------------------------------+
| Open-Meteo API: last 35 completed days                 |
+--------------------------------------------------------+
                             |
                             v
+--------------------------------------------------------+
| Disk cache: 1 API call per location per day            |
+--------------------------------------------------------+
                             |
                             v
+--------------------------------------------------------+
| Weekly features: rainfall x5 weeks,                    |
| temperature and humidity x2 weeks                      |
+--------------------------------------------------------+
                             |
                             v
+--------------------------------------------------------+
| Load saved model once, then predict                    |
+--------------------------------------------------------+
                             |
                             v
+--------------------------------------------------------+
| Risk tier + probabilities + recommended action         |
+--------------------------------------------------------+
```

**Live feature construction.** Only completed days are used (yesterday and earlier). Week 0 is the last 7 days, week 1 the 7 days before, up to week 4. Rainfall is summed per week; temperature and humidity are averaged. Inputs to the model (12 features):

| Group | Features |
|---|---|
| Ward (fixed, looked up) | `population_density_sqkm`, `water_logging_risk_index`, `sanitation_score` |
| Current week | `rainfall_mm`, `avg_temp_c`, `relative_humidity_pct` |
| Lags | `rainfall_mm_lag1w` to `lag4w`, `avg_temp_c_lag1w`, `relative_humidity_pct_lag1w` |

Lags are computed per `ward_id`, so one ward's history never leaks into another. This matches the 2-to-4-week mosquito breeding and incubation delay: rainfall 2 to 4 weeks back was the strongest predictor.

## 2. Target and risk tiers

The target is the **sum of dengue cases in the next 4 weeks** (t+1 to t+4) per ward. Tier cutoffs are the 50th, 75th and 90th percentiles of the training years only, so no test information is used:

| Tier | Cases in next 4 weeks | Share of rows |
|---|---|---|
| Low | 0-1 | about 50% |
| Medium | 2-19 | about 26% |
| High | 20-37 | about 15% |
| Critical | 38 or more | about 9% |

Critical is deliberately rare and occurs only in June-September, so it works as an alarm. The first 4 and last 4 weeks of each ward are dropped (no lags or no future), leaving 30,840 rows.

## 3. Chronological validation split

- **Final split:** train on 2016-2023 (about 24,800 rows), test on 2024-2025 (6,060 rows). The model never sees later weeks than it is tested on, which mirrors real use: forecasting the future.
- **Expanding-window validation:** for each test year 2020-2025, train on all earlier years. Macro-F1 ranged from 0.69 to 0.76 (mean 0.72), so performance is stable across years.
- **Why not a random split:** the target for week t overlaps heavily with weeks t+1 to t+3 of the same ward, so a random split puts near-duplicates in both sets. It scored accuracy 0.854 / macro-F1 0.805, against 0.779 / 0.689 on the chronological split. We report the chronological numbers.
- The saved model is refit on all years after validation.

## 4. Model selection

All models used the same 12 features and the same chronological split (test 2024-2025):

| Model | Accuracy | Macro-F1 | Critical recall | Saved size |
|---|---|---|---|---|
| Logistic regression | 0.781 | 0.675 | 51% | under 0.01 MB |
| Random forest (balanced) | 0.787 | 0.699 | 66% | about 15-95 MB |
| **HistGradientBoosting (balanced)** | 0.779 | 0.689 | 65% | **about 2 MB** |

**Chosen: HistGradientBoostingClassifier with balanced class weights.** Random forest and gradient boosting are statistically tied on accuracy and macro-F1 (each wins in 3 of 6 validation years). Gradient boosting was chosen for **size and speed**: a file of about 2 MB loads instantly, is safe to commit to GitHub, and predicts a single row in about 4 ms. It handles non-linear weather interactions (for example, the effect of rainfall depends on ward water-logging) that logistic regression misses, and needs no feature scaling. Balanced class weights raise recall on the rare High and Critical tiers, where a missed outbreak is costliest. 99.5% of predictions are within one tier of the truth; most errors are between neighbouring tiers, and the model tends to over-warn rather than under-warn.

Features dropped: `max_temp_c` and `min_temp_c` (correlation about 0.97 with `avg_temp_c`; removing them cost about 2.5% error). Ward features were kept: they separate high- and low-vulnerability wards (water-logging has the strongest ward-level link to cases).

## 5. Efficiency and cost controls

- **No LLM or paid API at inference, so token cost is zero.** The only external call is the free Open-Meteo weather API.
- **API minimisation:** results are cached on disk per (location, date), so at most 1 call per location per day, and repeated predictions make none. Coordinates are rounded to 2 decimals so neighbouring wards share a cache entry.
- **One request returns everything:** daily temperature and rainfall plus hourly humidity for 35 days in a single call.
- **Fast inference:** the serialized model is loaded once per process, and feature construction is a few vectorised operations.

## 6. Resilience

- Requests use a 10-second timeout and 3 retries with exponential backoff (1 s, 2 s, 4 s).
- If the API is down, the app falls back to the most recent cached data for that location and shows a clear "stale data" warning. If none exists, it shows an error instead of guessing.
- Weather data is validated before use (35 complete days, no missing values).
- The model's feature order is stored with the model, and probabilities are re-ordered Low to Critical, because scikit-learn returns classes alphabetically.

## 7. Limitations

- The output is one tier for the whole 28-day window (the model's target), not a day-by-day forecast.
- The dataset is treated as weekly aggregates (rainfall summed, temperature and humidity averaged); live data is aggregated the same way, but the dataset's week alignment is undocumented.
- Training weather varies per ward, while the live API supplies location-level grid values; this is a small mismatch.
