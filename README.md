# Dengue Outbreak Early Warning

**EDU AI Engineering Hackathon 2026, Phase 1 challenge: AI for Urban Dengue Outbreak**

A web dashboard that predicts dengue outbreak risk for a selected ward in Bangladesh (Dhaka North, Dhaka South, Chattogram, Gazipur, Narayanganj). Pick a city and ward number; the app fetches the last 35 days of weather for that ward, builds multi-week rainfall, temperature and humidity lag features, and shows a risk tier (**Low / Medium / High / Critical**) for the next 28 days, with the main weather drivers and a recommended action.

See ARCHITECTURE.md for the data flow, chronological validation split and model selection.

## Tech stack

- Python 3.12 or 3.13
- pandas, NumPy: data processing and lag features
- scikit-learn 1.6.1: HistGradientBoostingClassifier
- joblib: model serialization
- Streamlit and Altair: dashboard
- requests: weather API client
- Open-Meteo: free weather API (no API key needed)

## Setup and run (about 3 minutes)

Requires Python 3.12 or 3.13 (scikit-learn 1.6.1 has no prebuilt package for Python 3.14).

**Windows (PowerShell):**

```powershell      

git clone https://github.com/azambabar2002-cloud/hackathon1.git
cd hackathon1
py -3.12 -m venv .venv
.venv\Scripts\Activate
pip install -r requirements.txt
streamlit run app/dashboard.py
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first.


**Using the app:** choose a city, choose a ward number, and click **Predict**. The date is set to today automatically. No API key or environment variables are required.

## Project structure

```
data/ward_static.csv        ward coordinates and fixed ward features
data/                       raw training dataset
models/dengue_model.joblib  trained model
src/weather.py              Open-Meteo client with caching, retries and fallback
src/features.py             builds the 12 model features from daily weather
src/predict.py              loads the model, returns tier and probabilities
src/train.py                training script
app/dashboard.py            Streamlit dashboard
ARCHITECTURE.md             architecture and design decisions
```

## How it works (short)

1. The ward's population density, water-logging risk and sanitation score are looked up from `data/ward_static.csv`.
2. The app requests 35 days of daily temperature, rainfall and humidity from Open-Meteo (cached per location per day).
3. Weather is grouped into weekly blocks: rainfall for the last 5 weeks, temperature and humidity for the last 2 weeks.
4. The saved model predicts the dengue risk tier for the next 28 days.

## AI tools and APIs disclosure

| Tool / API | What it was used for |
|---|---|
| Claude (Anthropic) | Dataset analysis, feature and model selection guidance, preprocessing and Streamlit code drafts, and ARCHITECTURE.md / README.md drafting. All code was reviewed and is understood by the team. |
| Open-Meteo API (https://open-meteo.com) | Live and recent daily weather (temperature, rainfall, humidity) used as model input at prediction time. |

No LLM or paid API is called when the app runs; the only external call is the free Open-Meteo weather API.

## Attributions

Open-source libraries only (pandas, NumPy, scikit-learn, Streamlit, Altair, requests, joblib). The training dataset was provided by the hackathon organizers. No third-party code snippets were copied.

