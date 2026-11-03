# Real Estate Investment Advisor

A Streamlit dashboard for exploring Indian housing data, comparing properties with city-level market benchmarks, scoring investment potential, and running a transparent five-year appreciation scenario.

## Project structure

```text
real_estate_investment_advisor/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── india_housing_prices.csv
```

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Create a new GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, `.gitignore`, and the `data` folder.
3. In Streamlit Cloud, choose **New app** and select the repository.
4. Set the main file to `app.py` and deploy.

## What the dashboard includes

- Sidebar filters for state, property type, price, BHK, and rule-based investment status.
- Market overview charts for city prices, property type pricing, size versus price, and city investment rates.
- Investment advisor with a transparent 6-point rule-based score.
- Five-year price scenario using `Current Price × (1 + annual appreciation)^years`.
- Classification-model evaluation using a logistic-regression pipeline with one-hot encoding and scaling.

## Important modeling note

The included project copy contains a representative 100,000-row sample of the supplied dataset to keep the repository lightweight. The supplied data contains current property attributes and prices, but it does not contain an observed future-price column or a pre-existing `Good Investment` label. Therefore:

- `Good_Investment` is created transparently from domain rules: price per sq ft at or below the city median, plus supporting factors such as transport, parking, security, readiness, and property age.
- The five-year estimate is a scenario forecast, not a learned historical outcome. The annual appreciation rate is adjustable in the dashboard.
- The classifier is included to demonstrate the ML workflow. Its metrics measure how well it reproduces the rule-based label, not actual investment returns.

For a production-grade forecasting model, add historical transactions or a true future-price target and retrain the regression model against that observed outcome.
