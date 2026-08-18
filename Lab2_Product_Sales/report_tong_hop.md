# BÁO CÁO PRACTICE 02 — PREDICTING PRODUCT SALES

## 1. Mục tiêu

Xây dựng pipeline regression để dự đoán doanh số sản phẩm cho e-commerce, phục vụ inventory management và marketing strategy. Practice 02 yêu cầu Data Preparation, Feature Engineering, Model Selection, Training, Evaluation và Deployment. fileciteturn0file0L2-L39

## 2. Dataset

Project dùng **synthetic e-commerce product sales dataset**, tạo bằng `data/raw/generate_dataset.py` với seed `42`, gồm 6.000 observations và 14 columns.

| Tiêu chí | Đánh giá | Giải trình |
|---|---|---|
| Relevance | Đạt | Có product, customer, purchase history, marketing và seasonal variables; target `product_sales` liên tục. |
| Quality | Đạt | Có missing/outlier có chủ đích để minh họa data preparation; pipeline kiểm tra duplicate, xử lý IQR outlier và imputation. |
| Size | Đạt | 6.000 observations phù hợp train/test, comparison và tuning trong môi trường lab. |
| Features | Đạt | Có numeric, categorical và temporal features; tạo thêm popularity, CLV, seasonal index và ad efficiency. |

Chi tiết: `dataset_justification.md`.

## 3. Data Preparation

- Kiểm tra shape, schema, missing values, duplicates và descriptive statistics.
- Split train/test 80/20 với `random_state=42`.
- Feature engineering học medians từ train set.
- Numerical: median imputation + StandardScaler.
- Categorical: most-frequent imputation + One-Hot Encoding.
- Outlier: IQR clipping với bounds học từ training set.

Pipeline chỉ fit trên train để tránh data leakage.

## 4. EDA

EDA script tạo các đồ thị: sales distribution, sales by category, numeric correlation matrix và monthly sales trend.

## 5. Feature Engineering

Tạo `month`, `quarter`, `day_of_week`, `seasonal_index`, `product_popularity`, `customer_lifetime_value` và `ad_efficiency`, bám yêu cầu product popularity, CLV và seasonal trends của đề. fileciteturn0file0L18-L22

## 6. Model Selection

So sánh:

- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Ridge được tuning bằng GridSearchCV 5-fold với nhiều `alpha` và chọn theo RMSE thấp nhất.

## 7. Kết quả phiên bản hiện tại

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| **Tuned Ridge** | **14.487** | **332.030** | **18.222** | **0.7190** |
| Ridge | 14.491 | 332.042 | 18.222 | 0.7190 |
| Linear Regression | 14.493 | 332.085 | 18.223 | 0.7190 |
| Gradient Boosting | 15.332 | 375.857 | 19.387 | 0.6820 |
| Random Forest | 15.613 | 389.299 | 19.731 | 0.6706 |
| Decision Tree | 18.816 | 554.397 | 23.546 | 0.5309 |

**Best model:** Tuned Ridge, `alpha=10`.

## 8. Deployment

Model và preprocessing được serialize bằng Joblib. Streamlit cho phép nhập scenario mới và trả về predicted product sales; đây là bước deployment được yêu cầu trong đề. fileciteturn0file0L37-L39

## 9. Kết luận

Project hoàn thành pipeline: Data Preparation → Feature Engineering → Model Comparison → Hyperparameter Tuning → Evaluation → Deployment. Dataset được ghi rõ là synthetic để bảo đảm tính minh bạch và tái lập.

## 10. Hạn chế và hướng phát triển

Dataset hiện là synthetic. Với dữ liệu thực tế cần temporal validation, kiểm tra drift và business impact; đề cũng gợi ý time-series forecasting khi dữ liệu có thành phần thời gian. fileciteturn0file0L40-L46
