ARCHITECTURE: DENGUE OUTBREAK EARLY WARNING

A decision-support tool for public health officers in Bangladesh. It forecasts dengue cases for every ward for the next 4 weeks, turns the forecast into supplies, beds and spraying crews, shares limited stock fairly, and produces send-ready notices.

Live app: https://hackathon1-wpmz8tfch8ug5bgb39itin.streamlit.app/
Coverage: 60 wards in 5 city corporations (Chattogram, Dhaka North, Dhaka South, Gazipur, Narayanganj)


1. SYSTEM OVERVIEW

Pipeline, in order:
1. Open-Meteo weather, or recorded 2025 data
2. features.py - 14 model inputs
3. ward_static.csv - density, water-logging, sanitation
4. predict.py - 4 weekly models
5. cases_to_supplies.csv - saline, kits, beds
6. allocation.py - shares limited stock
7. overflow.py - bed surge routing
8. explain.py - why this forecast
9. simulate.py - what-if spraying crews
10. dispatch.py - SMS, WhatsApp, HTML, PDF
11. dashboard.py + ui.py - Streamlit app
12. exports/ - sample notices and briefs

The whole pipeline runs offline after the model and data files are loaded. The only network call is one weather request per city per day.


2. DATA FLOW, STEP BY STEP

1. Weather in. weather.py fetches the last 35 completed days of rain, temperature and humidity from Open-Meteo for the city centre. Results are cached on disk per location and day, retried 3 times, and fall back to the last good file if the service is down. In Recorded data mode the app replays each ward's own saved 2025 weather and needs no internet.

2. Features. features.py builds 14 inputs per ward: weekly rainfall now and 1 to 4 weeks ago, temperature and humidity now and 1 week ago, week-of-year as sine and cosine, plus the ward's 3 fixed traits (population density, water-logging risk, sanitation score).

3. Forecast. predict.py loads one saved gradient-boosting model per forecast week (weeks 1 to 4) and predicts expected cases for each ward and week.

4. Risk tier. Weekly cases map to Low (0 to 1), Medium (2 to 6), High (7 to 11) or Critical (12 or more).

5. Supplies. Expected cases are converted to saline bags, test kits and hospital beds with the dataset's own cases_to_supplies.csv lookup table.

6. Allocation (week 1). allocation.py shares the officer's weekly stock across wards (rules in section 4).

7. Bed overflow (week 1). overflow.py finds wards that need more beds than they have and routes the overflow to the nearest ward with room, in the same city.

8. Dispatch. dispatch.py fills fixed templates from the allocation and overflow results. No LLM and no external API is involved.

9. Explain and simulate. explain.py shows what drives a forecast; simulate.py shows what spraying crews would change.


3. MODULES

- app/dashboard.py: Streamlit app: sidebar inputs, 9 tabs, all charts and tables
- src/ui.py: Theme, animated mosquito hero, city overview charts
- src/weather.py: Open-Meteo client with daily cache, retries and stale fallback
- src/features.py: Builds the 14 model inputs from 35 days of weather
- src/predict.py: Loads the 4 models, forecasts, tiers, supplies lookup
- src/explain.py: Exact Shapley values over 11 factor groups (2,048 what-if runs, about 50 ms)
- src/allocation.py: Priority scoring, round-based stock sharing, crews, shortfall bands
- src/overflow.py: Ward bed capacity, surge detection, nearest-room routing
- src/simulate.py: Counterfactual forecasts with fewer breeding sites, ROI per crew
- src/dispatch.py: SMS (160-char parts), WhatsApp text, HTML brief, PDF brief
- scripts/make_exports.py: Regenerates the files in exports/ with the real pipeline
- models/dengue_weekly_models.joblib: Saved models, feature order, labels, tier cut-offs
- data/: ward_static.csv, cases_to_supplies.csv, recorded_2025.csv


4. KEY DESIGN DECISIONS

- One model per forecast week, not one model rolled forward. Each model predicts its own week directly, so errors do not compound. Later weeks are less accurate because part of the rain that drives them has not fallen yet, and the app says so.

- Lagged weather. Mosquito eggs need 7 to 14 days to mature and the virus then needs time to incubate, so cases follow rain by 2 to 4 weeks. The models read 5 weekly rainfall totals.

- Fair, explainable allocation. Priority is 70% predicted cases and 30% population density. Stock goes out in rounds of up to 50% of each ward's need. If round 1 cannot cover everyone, stock is shared across all wards in proportion to priority and need, so no ward is left at zero. Spraying crews are whole units handed out one level at a time in priority order (Low 0, Medium 1, High 2, Critical 3). Every shortfall is flagged: Minor (up to 25%), Serious (up to 50%), Critical deficit (above 50%).

- Honest overflow. Overflow that no ward in the city can absorb is reported as a city-wide shortage, not hidden.

- Offline notices. Messages come from fixed templates, so they are deterministic, free to run and safe from invented content. SMS text is plain ASCII.

- Cheap to run. One weather request per city per day, models loaded once, no paid API at runtime.


5. ASSUMPTIONS AND LIMITS

- Bed capacity is not in the dataset. Assumed: beds in a ward = density / 10,000 x 1.0 x share currently free (at least 1). The officer can change both numbers.

- Spraying effect is not in the dataset. Assumed: one crew-day cuts effective water-logging risk by 6%, with diminishing returns. The officer can change it.

- Weather is per city, not per ward, in live mode. Wards differ by their fixed traits.

- Planning estimates only. The tool does not diagnose illness or replace local surveillance data. The officer confirms numbers and calls the facilities.

- Sending is not automated. The app prepares send-ready text; delivery needs the officer's phone or a GSM gateway.

- Access. The current deployed build has no login. Firebase login for officers was chosen as the access method and is the next step.


6. DEPLOYMENT

- Hosted on Streamlit Community Cloud from the main branch; main file app/dashboard.py.
- Dependencies in requirements.txt (scikit-learn pinned to the training version so the saved model loads).
- Every push to main redeploys automatically.
- The weather cache is ephemeral on Streamlit Cloud; Recorded data mode is the backup if the weather service is slow.


7. EXPORTS (/exports)

Generated by python scripts/make_exports.py from the real pipeline using recorded 2025 weather (default city Chattogram, forecast week from 25 Aug 2025, stock at 60% of need).

- sms_notices.txt: One SMS per ward, split into 160-character parts where needed
- whatsapp_notices.txt: Longer formatted notices ready to paste into WhatsApp
- commissioner_brief.html: Printable brief: city totals, deficit alerts, ward table, bed routes
- commissioner_brief.pdf: The same brief as a PDF


8. REPOSITORY LAYOUT

app/dashboard.py     src/*.py     scripts/make_exports.py
models/              data/        assets/        exports/
.streamlit/config.toml            requirements.txt
ARCHITECTURE.md


9. VERSION CONTROL

Each commit message starts with "version-onsite-final:" followed by a short description of that one change, so the history reads as a continuous log of the onsite work.


10. AI DISCLOSURE

- Tool used: Claude (Anthropic), as a development assistant.

- What it helped with: planning the approach (preprocessing, lag features, risk tiers, model choice), reviewing and writing code, the dashboard look and feel and the mosquito hero animation (src/ui.py), deployment guidance, and this document.

- What it did not do: it is not used at runtime. The deployed app has no LLM calls, no AI-token API and no generated text; forecasts come from the saved scikit-learn models and all notices come from fixed templates.

- Human responsibility: the team chose the data, the assumptions in section 5 and the allocation rules, ran and checked the code, and is responsible fo