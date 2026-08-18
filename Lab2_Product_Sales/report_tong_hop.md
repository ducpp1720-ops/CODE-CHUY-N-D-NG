# BÁO CÁO PRACTICE 02 — PREDICTING PRODUCT SALES

## 1. Mục tiêu
Xây dựng mô hình hồi quy để dự đoán doanh số sản phẩm cho doanh nghiệp thương mại điện tử, phục vụ quản lý tồn kho và chiến lược marketing.

## 2. Dataset
Project sử dụng synthetic e-commerce product sales dataset, được tạo bằng `data/raw/generate_dataset.py`. Đề Practice 02 không cung cấp dataset bắt buộc; tài liệu gợi ý Kaggle và yêu cầu đánh giá theo Relevance, Quality, Size, Features. Vì vậy nguồn dữ liệu được ghi rõ là synthetic thay vì gọi nhầm là Kaggle.

Dataset có 6.000 observations, gồm product information, customer demographics, purchase history, marketing variables, seasonal/holiday variables và target `product_sales`.

| Tiêu chí | Đánh giá |
|---|---|
| Relevance | Đạt: các nhóm biến bám sát đề bài. |
| Quality | Đạt: có kiểm tra missing/duplicate/outlier và preprocessing. |
| Size | Đạt: 6.000 observations, đủ cho train/test và model comparison trong lab. |
| Features | Đạt: numeric, categorical, temporal và feature engineering. |

## 3. Data Preparation
- Kiểm tra shape, dtypes, missing values, duplicate và outlier.
- Numerical: median imputation + StandardScaler.
- Categorical: most-frequent imputation + One-Hot Encoding.
- Train/test split 80/20, random_state=42.
- Preprocessor chỉ fit trên train để tránh data leakage.

## 4. Feature Engineering
Tạo `month`, `quarter`, `day_of_week`, `seasonal_index`, `product_popularity`, `customer_lifetime_value`, `ad_efficiency`.

## 5. Model Selection
So sánh Linear Regression, Ridge, Decision Tree, Random Forest và Gradient Boosting; sau đó tuning Ridge.

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Tuned Ridge | 14.393 | 324.924 | 18.026 | 0.725 |
| Ridge | 14.400 | 324.981 | 18.027 | 0.725 |
| Linear Regression | 14.404 | 325.035 | 18.029 | 0.725 |
| Gradient Boosting | 15.327 | 373.884 | 19.336 | 0.684 |
| Random Forest | 15.538 | 384.385 | 19.606 | 0.675 |
| Decision Tree | 18.355 | 537.274 | 23.179 | 0.545 |

Best model: **Tuned Ridge Regression, alpha=10**.

## 6. Evaluation
Metric chính gồm MAE, MSE, RMSE và R². RMSE được ưu tiên khi so sánh vì cùng đơn vị với target và phạt sai số lớn mạnh hơn MAE.

## 7. Deployment
Model và preprocessing được serialize bằng Joblib. Streamlit cung cấp giao diện nhập scenario mới và trả về predicted sales.

## 8. Kết luận
Pipeline hoàn chỉnh: Data Preparation → Feature Engineering → Model Selection → Training → Evaluation → Deployment. Trên dataset lab, Tuned Ridge đạt RMSE khoảng 18.03 và R² khoảng 0.725.
