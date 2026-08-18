import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create lab-required features and clip price outliers using train-set statistics."""
    def fit(self, X, y=None):
        X = X.copy()
        price = pd.to_numeric(X["price"], errors="coerce")
        q1, q3 = price.quantile(.25), price.quantile(.75)
        iqr = q3 - q1
        self.price_lower_ = float(q1 - 1.5*iqr)
        self.price_upper_ = float(q3 + 1.5*iqr)
        self.price_median_ = float(price.median())
        self.rating_median_ = float(pd.to_numeric(X["rating"], errors="coerce").median())
        self.purchase_frequency_median_ = float(pd.to_numeric(X["purchase_frequency"], errors="coerce").median())
        return self

    def transform(self, X):
        X = X.copy()
        X["purchase_date"] = pd.to_datetime(X["purchase_date"])
        X["price"] = pd.to_numeric(X["price"], errors="coerce").clip(self.price_lower_, self.price_upper_)
        X["month"] = X["purchase_date"].dt.month
        X["quarter"] = X["purchase_date"].dt.quarter
        X["day_of_week"] = X["purchase_date"].dt.dayofweek
        X["seasonal_index"] = 1 + .15*np.sin(2*np.pi*X["month"]/12) + .12*X["is_holiday"].fillna(0)
        X["product_popularity"] = np.log1p(X["number_of_reviews"].fillna(0))*X["rating"].fillna(self.rating_median_)
        X["customer_lifetime_value"] = X["previous_purchases"].fillna(0)*X["price"].fillna(self.price_median_)*X["purchase_frequency"].fillna(self.purchase_frequency_median_)
        X["ad_efficiency"] = X["ad_spend"].fillna(0)/(X["campaigns_run"].fillna(0)+1)
        return X.drop(columns=["purchase_date"])
