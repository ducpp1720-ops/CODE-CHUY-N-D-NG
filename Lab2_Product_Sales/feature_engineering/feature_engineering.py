import joblib
import pandas as pd
import sys
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_engineering_components import FeatureEngineer, IQRClipper


df = pd.read_csv(ROOT / "data/raw/product_sales.csv")
target = "product_sales"
X = df.drop(columns=[target])
y = df[target]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric = [
    "price", "customer_age", "previous_purchases", "purchase_frequency",
    "campaigns_run", "ad_spend", "number_of_reviews", "rating",
    "is_holiday", "month", "quarter", "day_of_week", "seasonal_index",
    "product_popularity", "customer_lifetime_value", "ad_efficiency",
]
categorical = ["product_category", "gender", "location"]

pre = Pipeline([
    ("feature_engineering", FeatureEngineer()),
    ("outlier_clipping", IQRClipper(columns=numeric)),
    ("preprocessing", ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ),
    ])),
])

# Fit only on training data to prevent data leakage.
Xtr_t = pre.fit_transform(Xtr, ytr)
Xte_t = pre.transform(Xte)

joblib.dump(pre, ROOT / "data/processed/preprocessor.pkl")
joblib.dump(
    {"X_train": Xtr_t, "X_test": Xte_t, "y_train": ytr, "y_test": yte},
    ROOT / "data/processed/split.pkl",
)
print("Train:", Xtr_t.shape, "Test:", Xte_t.shape)
