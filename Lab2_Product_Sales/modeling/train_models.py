import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

obj = joblib.load(ROOT / "data/processed/split.pkl")
Xtr, Xte, ytr, yte = obj["X_train"], obj["X_test"], obj["y_train"], obj["y_test"]

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10, min_samples_leaf=3),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=80, max_depth=14, min_samples_leaf=2, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=3, learning_rate=0.05, loss="huber"),
}

rows = []
for name, model in models.items():
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mse = mean_squared_error(yte, pred)
    rows.append([name, mean_absolute_error(yte, pred), mse, mse ** 0.5, r2_score(yte, pred)])

# Tune Ridge with 5-fold cross-validation as required by the lab workflow.
ridge_grid = GridSearchCV(
    Ridge(),
    {"alpha": [0.01, 0.1, 1, 10, 50, 100]},
    scoring="neg_root_mean_squared_error",
    cv=5,
    n_jobs=-1,
)
ridge_grid.fit(Xtr, ytr)
pred = ridge_grid.best_estimator_.predict(Xte)
mse = mean_squared_error(yte, pred)
rows.append(["Tuned Ridge", mean_absolute_error(yte, pred), mse, mse ** 0.5, r2_score(yte, pred)])

results = pd.DataFrame(rows, columns=["model", "MAE", "MSE", "RMSE", "R2"]).sort_values("RMSE")
results.to_csv(ROOT / "modeling/model_comparison.csv", index=False)

# Save the best fitted estimator and metrics.
best_row = results.iloc[0]
best_model = ridge_grid.best_estimator_ if best_row["model"] == "Tuned Ridge" else models[best_row["model"]]
joblib.dump(best_model, ROOT / "modeling/best_model.pkl")
metrics = {
    "model": str(best_row["model"]),
    "MAE": float(best_row["MAE"]),
    "MSE": float(best_row["MSE"]),
    "RMSE": float(best_row["RMSE"]),
    "R2": float(best_row["R2"]),
    "best_params": ridge_grid.best_params_,
}
with open(ROOT / "modeling/best_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# Evaluation plots.
plt.figure(figsize=(7, 6))
plt.scatter(yte, pred, alpha=0.35)
lo, hi = min(yte.min(), pred.min()), max(yte.max(), pred.max())
plt.plot([lo, hi], [lo, hi])
plt.xlabel("Actual sales")
plt.ylabel("Predicted sales")
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.savefig(FIG / "actual_vs_predicted.png", dpi=220)
plt.close()

residuals = yte - pred
plt.figure(figsize=(8, 5))
plt.scatter(pred, residuals, alpha=0.35)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted sales")
plt.ylabel("Residual")
plt.title("Residual Analysis")
plt.tight_layout()
plt.savefig(FIG / "residual_analysis.png", dpi=220)
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(results["model"], results["RMSE"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("RMSE")
plt.title("Model Comparison")
plt.tight_layout()
plt.savefig(FIG / "model_comparison.png", dpi=220)
plt.close()

print(results.to_string(index=False))
print("\nBest:", metrics)
