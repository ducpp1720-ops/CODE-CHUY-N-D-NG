import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "feature_engineering"))

pre = joblib.load(ROOT / "data/processed/preprocessor.pkl")
model = joblib.load(ROOT / "modeling/best_model.pkl")

st.set_page_config(page_title="Product Sales Predictor", page_icon="📈")
st.title("📈 Product Sales Prediction")
st.caption("Practice 02 - Machine Learning Regression")

with st.form("scenario"):
    purchase_date = st.date_input("Purchase date")
    category = st.selectbox("Product category", ["Electronics","Home","Beauty","Sports","Fashion","Books"])
    price = st.number_input("Price", min_value=1.0, value=60.0)
    age = st.number_input("Customer age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male","Female","Other"])
    location = st.selectbox("Location", ["North","Central","South"])
    previous = st.number_input("Previous purchases", min_value=0, value=8)
    frequency = st.number_input("Purchase frequency", min_value=.1, value=4.0)
    campaigns = st.number_input("Campaigns run", min_value=0, value=2)
    ad_spend = st.number_input("Ad spend", min_value=0.0, value=250.0)
    reviews = st.number_input("Number of reviews", min_value=0, value=80)
    rating = st.number_input("Rating", min_value=1.0, max_value=5.0, value=4.2)
    holiday = st.selectbox("Holiday", [0,1])
    submitted = st.form_submit_button("Predict sales")

if submitted:
    row = pd.DataFrame([{
        "purchase_date": str(purchase_date), "product_category": category, "price": price,
        "customer_age": age, "gender": gender, "location": location,
        "previous_purchases": previous, "purchase_frequency": frequency,
        "campaigns_run": campaigns, "ad_spend": ad_spend, "number_of_reviews": reviews,
        "rating": rating, "is_holiday": holiday,
    }])
    prediction = float(model.predict(pre.transform(row))[0])
    st.metric("Predicted Product Sales", f"{prediction:.2f}")
