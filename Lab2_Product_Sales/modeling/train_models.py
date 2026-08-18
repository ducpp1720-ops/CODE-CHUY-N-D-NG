import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
obj = joblib.load(ROOT / "data/processed/split.pkl")
Xtr, Xte, ytr, yte = obj["X_train"], obj["X_test"], obj["y_train"], obj["y_test"]
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=8),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200, max_depth=12, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=150, max_depth=3),
}
results = []
for name, model in models.items():
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mse = mean_squared_error(yte, pred)
    results.append({"model":name,"MAE":mean_absolute_error(yte,pred),"MSE":mse,"RMSE":mse**.5,"R2":r2_score(yte,pred)})

grid = GridSearchCV(Ridge(), {"alpha":[.01,.1,1,10,50,100]}, scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
grid.fit(Xtr, ytr)
pred = grid.predict(Xte)
mse = mean_squared_error(yte, pred)
results.append({"model":"Tuned Ridge","MAE":mean_absolute_error(yte,pred),"MSE":mse,"RMSE":mse**.5,"R2":r2_score(yte,pred)})
results = sorted(results, key=lambda x:x["RMSE"])
pd.DataFrame(results).to_csv(ROOT/"modeling/model_comparison.csv", index=False)
joblib.dump(grid.best_estimator_, ROOT/"modeling/best_model.pkl")
with open(ROOT/"modeling/best_metrics.json","w",encoding="utf-8") as f:
    json.dump({**results[0], "best_params":grid.best_params_}, f, indent=2)
print(pd.DataFrame(results).to_string(index=False))
