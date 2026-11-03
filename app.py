from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "data" / "india_housing_prices.csv"


def money_lakhs(value: float) -> str:
    return f"₹{value:,.2f} L"


@st.cache_data(show_spinner="Loading housing data…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Amenities_Count"] = df["Amenities"].fillna("").apply(
        lambda x: len([item for item in str(x).split(",") if item.strip()])
    )
    df["Has_Parking"] = (df["Parking_Space"].str.lower() == "yes").astype(int)
    df["Has_Security"] = (df["Security"].str.lower() == "yes").astype(int)
    df["Transport_Score"] = df["Public_Transport_Accessibility"].map(
        {"Low": 1, "Medium": 2, "High": 3}
    ).fillna(1)
    df["Is_Ready"] = (df["Availability_Status"] == "Ready_to_Move").astype(int)
    city_median = df.groupby("City")["Price_per_SqFt"].transform("median")
    df["City_Median_Price_per_SqFt"] = city_median
    df["Value_vs_City_Median"] = df["Price_per_SqFt"] / city_median.replace(0, np.nan)
    df["Investment_Score"] = (
        (df["Value_vs_City_Median"] <= 1).astype(int)
        + (df["Transport_Score"] >= 2).astype(int)
        + df["Has_Parking"]
        + df["Has_Security"]
        + df["Is_Ready"]
        + (df["Age_of_Property"] <= 15).astype(int)
    )
    # This is a transparent, domain-rule label because the supplied data has no label.
    df["Good_Investment"] = (
        (df["Value_vs_City_Median"] <= 1.0) & (df["Investment_Score"] >= 4)
    ).astype(int)
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "State", "City", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
        "Price_per_SqFt", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
        "Public_Transport_Accessibility", "Parking_Space", "Security", "Furnished_Status",
        "Facing", "Owner_Type", "Availability_Status", "Amenities_Count", "Transport_Score",
        "Value_vs_City_Median",
    ]
    return df[cols].copy()


@st.cache_resource(show_spinner="Training investment classifier…")
def train_classifier(df: pd.DataFrame):
    # A capped sample keeps Streamlit Cloud startup responsive while retaining broad coverage.
    sample = df.sample(min(60000, len(df)), random_state=42)
    X = make_features(sample)
    y = sample["Good_Investment"]
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )
    model = Pipeline(
        [("preprocess", prep), ("classifier", LogisticRegression(max_iter=300, class_weight="balanced"))]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred, zero_division=0),
        "matrix": confusion_matrix(y_test, pred),
        "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    }
    return model, metrics


def investment_recommendation(row: dict, df: pd.DataFrame) -> dict:
    city_median = float(df.loc[df["City"] == row["City"], "Price_per_SqFt"].median())
    value_ratio = row["Price_per_SqFt"] / city_median if city_median else 1.0
    score = sum(
        [
            value_ratio <= 1,
            row["Transport_Score"] >= 2,
            row["Parking_Space"] == "Yes",
            row["Security"] == "Yes",
            row["Availability_Status"] == "Ready_to_Move",
            row["Age_of_Property"] <= 15,
        ]
    )
    return {
        "score": int(score),
        "max_score": 6,
        "city_median": city_median,
        "value_ratio": value_ratio,
        "good": bool(value_ratio <= 1 and score >= 4),
    }


def filtered_insights(df: pd.DataFrame):
    left, right = st.columns(2)
    with left:
        by_city = df.groupby("City", as_index=False)["Price_in_Lakhs"].mean().sort_values("Price_in_Lakhs", ascending=False).head(15)
        st.plotly_chart(
            px.bar(by_city, x="Price_in_Lakhs", y="City", orientation="h", title="Average price by city", color="Price_in_Lakhs", color_continuous_scale="Blues"),
            use_container_width=True,
        )
    with right:
        by_type = df.groupby("Property_Type", as_index=False)["Price_per_SqFt"].mean().sort_values("Price_per_SqFt", ascending=False)
        st.plotly_chart(
            px.bar(by_type, x="Property_Type", y="Price_per_SqFt", title="Average price per sq ft by property type", color="Property_Type"),
            use_container_width=True,
        )
    left, right = st.columns(2)
    with left:
        sample = df.sample(min(12000, len(df)), random_state=7)
        st.plotly_chart(
            px.scatter(sample, x="Size_in_SqFt", y="Price_in_Lakhs", color="Property_Type", opacity=0.45, hover_data=["City", "BHK"], title="Size versus price"),
            use_container_width=True,
        )
    with right:
        inv = df.groupby("City", as_index=False)["Good_Investment"].mean().sort_values("Good_Investment", ascending=False).head(15)
        inv["Good_Investment"] *= 100
        st.plotly_chart(
            px.bar(inv, x="Good_Investment", y="City", orientation="h", title="Good-investment rate by city (%)", color="Good_Investment", color_continuous_scale="Teal"),
            use_container_width=True,
        )


df = load_data()

st.markdown("# Real Estate Investment Advisor")
st.markdown("Use data-backed filters, investment scoring, and scenario-based five-year price forecasts to explore Indian housing opportunities.")

with st.sidebar:
    st.header("Explore the dataset")
    states = st.multiselect("State", sorted(df["State"].unique()), default=[])
    property_types = st.multiselect("Property type", sorted(df["Property_Type"].unique()), default=[])
    min_price, max_price = st.slider("Price range (₹ lakhs)", 0.0, 500.0, (0.0, 500.0), 5.0)
    min_bhk, max_bhk = st.slider("BHK", 1, 5, (1, 5))
    only_good = st.checkbox("Show only rule-based good investments", value=False)

filtered = df[
    df["Price_in_Lakhs"].between(min_price, max_price)
    & df["BHK"].between(min_bhk, max_bhk)
]
if states:
    filtered = filtered[filtered["State"].isin(states)]
if property_types:
    filtered = filtered[filtered["Property_Type"].isin(property_types)]
if only_good:
    filtered = filtered[filtered["Good_Investment"] == 1]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Properties in view", f"{len(filtered):,}")
k2.metric("Median price", money_lakhs(filtered["Price_in_Lakhs"].median()) if len(filtered) else "—")
k3.metric("Median price / sq ft", f"₹{filtered['Price_per_SqFt'].median():.2f}" if len(filtered) else "—")
k4.metric("Good-investment rate", f"{filtered['Good_Investment'].mean() * 100:.1f}%" if len(filtered) else "—")

if filtered.empty:
    st.warning("No properties match the current filters. Adjust the sidebar selections.")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["Market overview", "Investment advisor", "Price forecast", "Model evaluation"])
    with tab1:
        st.subheader("Market overview")
        filtered_insights(filtered)
        st.subheader("Data preview")
        st.dataframe(filtered.head(100), use_container_width=True, height=320)

    with tab2:
        st.subheader("Is this property a good investment?")
        st.caption("The recommendation combines price versus the selected city's median price per sq ft with transport, parking, security, readiness, and property age.")
        c1, c2, c3 = st.columns(3)
        with c1:
            city = st.selectbox("City", sorted(df["City"].unique()), index=0)
            city_rows = df[df["City"] == city]
            property_type = st.selectbox("Property type", sorted(df["Property_Type"].unique()))
            state = st.selectbox("State", sorted(df["State"].unique()))
            bhk = st.slider("BHK", 1, 5, 3)
            size = st.number_input("Size (sq ft)", 500, 5000, 1800, 100)
        with c2:
            price = st.number_input("Price (₹ lakhs)", 10.0, 500.0, 150.0, 5.0)
            ppsf = st.number_input("Price per sq ft", 0.0, 1.0, 0.10, 0.01)
            age = st.slider("Property age", 2, 35, 10)
            schools = st.slider("Nearby schools", 1, 10, 5)
            hospitals = st.slider("Nearby hospitals", 1, 10, 5)
        with c3:
            transport = st.selectbox("Public transport", ["Low", "Medium", "High"], index=2)
            parking = st.selectbox("Parking", ["No", "Yes"], index=1)
            security = st.selectbox("Security", ["No", "Yes"], index=1)
            furnished = st.selectbox("Furnished status", sorted(df["Furnished_Status"].unique()))
            availability = st.selectbox("Availability", sorted(df["Availability_Status"].unique()))

        row = {
            "State": state, "City": city, "Property_Type": property_type, "BHK": bhk,
            "Size_in_SqFt": size, "Price_in_Lakhs": price, "Price_per_SqFt": ppsf,
            "Age_of_Property": age, "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
            "Public_Transport_Accessibility": transport, "Parking_Space": parking, "Security": security,
            "Furnished_Status": furnished, "Facing": "North", "Owner_Type": "Owner",
            "Availability_Status": availability, "Amenities_Count": 3,
            "Transport_Score": {"Low": 1, "Medium": 2, "High": 3}[transport],
            "Value_vs_City_Median": ppsf / max(float(city_rows["Price_per_SqFt"].median()), 0.0001),
        }
        result = investment_recommendation(row, df)
        if result["good"]:
            st.success(f"Recommended: Good investment — score {result['score']}/{result['max_score']}")
        else:
            st.warning(f"Review carefully: score {result['score']}/{result['max_score']}")
        r1, r2, r3 = st.columns(3)
        r1.metric("City median price / sq ft", f"₹{result['city_median']:.2f}")
        r2.metric("Price vs city median", f"{result['value_ratio']:.0%}")
        r3.metric("Rule-based score", f"{result['score']}/{result['max_score']}")

    with tab3:
        st.subheader("Estimated price after five years")
        st.info("The supplied dataset has no observed future-price column. This forecast is therefore a scenario model: current price × (1 + annual appreciation)⁵. Adjust the assumption for sensitivity analysis.")
        current = st.number_input("Current property price (₹ lakhs)", 10.0, 500.0, float(filtered["Price_in_Lakhs"].median()), 5.0)
        appreciation = st.slider("Annual appreciation assumption", 0.0, 20.0, 8.0, 0.5) / 100
        years = st.slider("Forecast horizon (years)", 1, 10, 5)
        forecast = current * ((1 + appreciation) ** years)
        gain = forecast - current
        p1, p2, p3 = st.columns(3)
        p1.metric("Current price", money_lakhs(current))
        p2.metric(f"Estimated price after {years} years", money_lakhs(forecast), f"+{money_lakhs(gain)}")
        p3.metric("Total expected growth", f"{(forecast / current - 1) * 100:.1f}%")
        years_axis = list(range(0, years + 1))
        values = [current * ((1 + appreciation) ** year) for year in years_axis]
        chart = pd.DataFrame({"Year": years_axis, "Estimated Price (₹ lakhs)": values})
        st.plotly_chart(px.line(chart, x="Year", y="Estimated Price (₹ lakhs)", markers=True, title="Scenario forecast"), use_container_width=True)

    with tab4:
        st.subheader("Classification model evaluation")
        model, metrics = train_classifier(df)
        a, b = st.columns(2)
        a.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        b.metric("F1 score", f"{metrics['f1']:.3f}")
        st.write("Confusion matrix")
        st.dataframe(pd.DataFrame(metrics["matrix"], index=["Actual: Not good", "Actual: Good"], columns=["Predicted: Not good", "Predicted: Good"]))
        st.caption("The classifier learns the transparent rule-based label created from this dataset. It is useful for demonstrating the ML workflow, but it should not be interpreted as an independently observed investment outcome.")

st.divider()
st.caption("Built for the Real Estate Investment Advisor capstone • Dataset: india_housing_prices.csv • Use the README for GitHub and Streamlit Cloud deployment.")
