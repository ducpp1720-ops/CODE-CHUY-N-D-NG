# Lab Practice 02 – Predicting Product Sales

## Mục tiêu

Xây dựng pipeline Machine Learning hoàn chỉnh cho bài toán **regression dự đoán product sales** trong e-commerce. Project bám trực tiếp đề Practice 02: Data Preparation → Feature Engineering → Model Selection → Training → Evaluation → Deployment. fileciteturn0file0L2-L39

## Dataset

Project dùng **synthetic e-commerce dataset** tái tạo được bằng `data/raw/generate_dataset.py` với seed `42`.

- 6.000 observations, 14 columns
- Product: category, price, reviews, rating
- Customer: age, gender, location
- Purchase history: previous purchases, purchase frequency
- Marketing: campaigns run, ad spend
- Seasonal: purchase date, holiday indicator
- Target: `product_sales`

Chi tiết Relevance / Quality / Size / Features: [`dataset_justification.md`](dataset_justification.md).

## Pipeline

```text
Raw Dataset
 → EDA + Data Quality Check
 → Train/Test Split 80/20
 → Feature Engineering
 → IQR Outlier Clipping
 → Imputation + Scaling + One-Hot Encoding
 → Model Comparison
 → Ridge Hyperparameter Tuning (5-fold CV)
 → MAE / MSE / RMSE / R²
 → Save Best Model
 → Streamlit Deployment
```

## Feature Engineering

- `product_popularity`
- `customer_lifetime_value`
- `seasonal_index`
- `month`, `quarter`, `day_of_week`
- `ad_efficiency`

## Models

So sánh:

1. Linear Regression
2. Ridge Regression
3. Decision Tree Regressor
4. Random Forest Regressor
5. Gradient Boosting Regressor

Sau comparison, Ridge được tuning với `GridSearchCV` 5-fold trên nhiều giá trị `alpha`.

## Kết quả phiên bản hiện tại

- **Best model:** Tuned Ridge
- **alpha:** 10
- **MAE:** 14.487
- **MSE:** 332.030
- **RMSE:** 18.222
- **R²:** 0.7190

Kết quả được sinh tự động trong `modeling/model_comparison.csv` và `modeling/best_metrics.json`.

## Chạy project

```powershell
cd Lab2_Product_Sales
pip install -r requirements.txt
python data/raw/generate_dataset.py
python eda/eda.py
python feature_engineering/feature_engineering.py
python modeling/train_models.py
python tests/smoke_test.py
streamlit run app_streamlit.py
```

## Tài liệu trả lab

- `report_tong_hop.md` — báo cáo tổng hợp
- `docs/QUESTIONS_AND_ANSWERS.md` — câu hỏi vấn đáp
- `dataset_justification.md` — giải trình lựa chọn dataset
- `chon_thuat_toan.md` — lý do chọn thuật toán
- `ML_flow.md` — luồng Machine Learning
- `deploy.md` — hướng dẫn deployment
- `tailieu/source_reference.md` — nguồn đề và tài liệu tham chiếu

## Reproducibility

Dataset và artifacts có thể tái tạo từ source. Preprocessing chỉ fit trên training split để tránh data leakage. Outlier bounds và median fallback cũng được học từ train set.
