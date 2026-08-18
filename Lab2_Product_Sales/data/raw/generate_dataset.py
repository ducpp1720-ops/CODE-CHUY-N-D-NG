import pandas as pd
import numpy as np
from pathlib import Path

rng = np.random.default_rng(42)
n = 6000
dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
purchase_date = rng.choice(dates, n)
categories = rng.choice(["Electronics", "Home", "Beauty", "Sports", "Fashion", "Books"], n, p=[.18,.18,.14,.16,.20,.14])
regions = rng.choice(["North", "Central", "South"], n, p=[.35,.20,.45])
genders = rng.choice(["Male", "Female", "Other"], n, p=[.47,.50,.03])
age = rng.integers(18, 66, n)
price = np.round(np.exp(rng.normal(np.log(45), .75, n)), 2).clip(5, 500)
reviews = rng.poisson(80, n)
rating = np.round(np.clip(rng.normal(4.1, .45, n), 1, 5), 2)
prev = rng.poisson(8, n)
freq = np.round(np.clip(rng.gamma(2.2, 2, n), .1, 30), 2)
campaigns = rng.poisson(2.5, n)
ad = np.round(np.clip(rng.gamma(2, 120, n), 0, 1500), 2)
holiday = pd.Series(purchase_date).dt.month.isin([11, 12]).astype(int).to_numpy()
month = pd.Series(purchase_date).dt.month.to_numpy()
season = 1 + .15*np.sin(2*np.pi*month/12) + .12*holiday
cat = pd.Series(categories).map({"Electronics":35,"Home":20,"Beauty":12,"Sports":18,"Fashion":15,"Books":8}).to_numpy()
reg = pd.Series(regions).map({"North":4,"Central":0,"South":7}).to_numpy()
noise = rng.normal(0, 18, n)
sales = (30 + .075*price + .9*np.sqrt(reviews+1)*10 + 2.2*prev + 3.5*freq + 1.8*campaigns + .018*ad + 8*rating + cat + reg)*season + noise
sales = np.clip(sales, 1, None).round(2)

df = pd.DataFrame({
    "purchase_date": pd.to_datetime(purchase_date), "product_category": categories,
    "price": price, "customer_age": age, "gender": genders, "location": regions,
    "previous_purchases": prev, "purchase_frequency": freq, "campaigns_run": campaigns,
    "ad_spend": ad, "number_of_reviews": reviews, "rating": rating,
    "is_holiday": holiday, "product_sales": sales
})
for c in ["price", "customer_age", "rating", "location"]:
    idx = rng.choice(n, int(n*.01), replace=False)
    df.loc[idx, c] = np.nan
idx = rng.choice(n, 20, replace=False)
df.loc[idx, "price"] *= 8
out = Path(__file__).resolve().parent / "product_sales.csv"
df.to_csv(out, index=False)
print(df.shape)
