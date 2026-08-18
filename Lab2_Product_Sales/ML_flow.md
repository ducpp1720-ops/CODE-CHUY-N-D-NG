# ML Flow — Practice 02: Predicting Product Sales

`Define Problem → Data Selection → Data Preparation/EDA → Feature Engineering → Model Selection → Training → Evaluation → Deployment`

## Mapping to the assignment

1. Inspect schema, missing values, duplicates and outliers.
2. Split data 80/20 before fitting the preprocessing pipeline.
3. Treat outliers with train-set IQR clipping.
4. Handle numerical missing values with median imputation and categorical missing values with most-frequent imputation.
5. Encode categorical variables with One-Hot Encoding and standardize numerical variables.
6. Create product popularity, customer lifetime value, seasonal trends and ad efficiency features.
7. Compare Linear Regression, Ridge, Decision Tree, Random Forest and Gradient Boosting.
8. Tune **Ridge Regression** with `GridSearchCV` using 5-fold CV and a grid of `alpha` values.
9. Evaluate with MAE, MSE, RMSE and R² on the held-out test set.
10. Serialize the preprocessing pipeline and best model and deploy with Streamlit.

## Leakage control

Feature engineering medians, IQR bounds, imputers, scaler parameters and One-Hot Encoder are learned only from the training split. The test split is transformed afterward with the fitted pipeline.
