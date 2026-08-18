from pathlib import Path
import json
import sys
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pre = joblib.load(ROOT / "data/processed/preprocessor.pkl")
model = joblib.load(ROOT / "modeling/best_model.pkl")

row = pd.DataFrame([{
    "purchase_date": "2024-12-20",
    "product_category": "Electronics",
    "price": 60,
    "customer_age": 30,
    "gender": "Female",
    "location": "South",
    "previous_purchases": 8,
    "purchase_frequency": 4,
    "campaigns_run": 2,
    "ad_spend": 250,
    "number_of_reviews": 80,
    "rating": 4.2,
    "is_holiday": 1,
}])

X = pre.transform(row)
pred = float(model.predict(X)[0])
metrics = json.load(open(ROOT / "modeling/best_metrics.json", encoding="utf-8"))

assert X.shape[0] == 1
assert pred > 0
assert metrics["model"] == "Tuned Ridge"
assert metrics["best_params"]["alpha"] == 10
print(f"Smoke test passed. Prediction={pred:.2f}")
