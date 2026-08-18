import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from transformers import FeatureEngineer

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/raw/product_sales.csv")
X = df.drop(columns=["product_sales"])
y = df["product_sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
numeric = ["price","customer_age","previous_purchases","purchase_frequency","campaigns_run","ad_spend","number_of_reviews","rating","is_holiday","month","quarter","day_of_week","seasonal_index","product_popularity","customer_lifetime_value","ad_efficiency"]
categorical = ["product_category","gender","location"]
pre = Pipeline([("fe", FeatureEngineer()), ("prep", ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
]))])
Xtr = pre.fit_transform(X_train, y_train)
Xte = pre.transform(X_test)
joblib.dump(pre, ROOT / "data/processed/preprocessor.pkl")
joblib.dump({"X_train":Xtr,"X_test":Xte,"y_train":y_train,"y_test":y_test}, ROOT / "data/processed/split.pkl")
print("Train:", Xtr.shape, "Test:", Xte.shape)
