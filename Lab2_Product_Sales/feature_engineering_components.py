import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create domain features using train-only fallback medians."""

    def fit(self, X, y=None):
        X = X.copy()
        self.rating_median_ = pd.to_numeric(X["rating"], errors="coerce").median()
        self.price_median_ = pd.to_numeric(X["price"], errors="coerce").median()
        self.frequency_median_ = pd.to_numeric(
            X["purchase_frequency"], errors="coerce"
        ).median()
        return self

    def transform(self, X):
        X = X.copy()
        X["purchase_date"] = pd.to_datetime(X["purchase_date"])
        X["month"] = X["purchase_date"].dt.month
        X["quarter"] = X["purchase_date"].dt.quarter
        X["day_of_week"] = X["purchase_date"].dt.dayofweek

        holiday = X["is_holiday"].fillna(0)
        X["seasonal_index"] = 1 + 0.15 * np.sin(2 * np.pi * X["month"] / 12) + 0.12 * holiday

        rating = X["rating"].fillna(self.rating_median_)
        price = X["price"].fillna(self.price_median_)
        frequency = X["purchase_frequency"].fillna(self.frequency_median_)

        X["product_popularity"] = np.log1p(X["number_of_reviews"].fillna(0)) * rating
        X["customer_lifetime_value"] = X["previous_purchases"].fillna(0) * price * frequency
        X["ad_efficiency"] = X["ad_spend"].fillna(0) / (X["campaigns_run"].fillna(0) + 1)
        return X.drop(columns=["purchase_date"])


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip numerical features using IQR bounds learned from training data."""

    def __init__(self, columns=None, factor=1.5):
        self.columns = columns
        self.factor = factor

    def fit(self, X, y=None):
        X = X.copy()
        self.columns_ = list(self.columns or X.select_dtypes(include=np.number).columns)
        q1 = X[self.columns_].quantile(0.25)
        q3 = X[self.columns_].quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns_:
            X[col] = X[col].clip(self.lower_[col], self.upper_[col])
        return X
