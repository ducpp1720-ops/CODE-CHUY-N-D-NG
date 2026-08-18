# Questions & Answers — Practice 02

## WHY

### 1. Tại sao chọn Product Sales Prediction?
Đề yêu cầu dự đoán product sales để hỗ trợ inventory management và marketing strategy. Target là biến số liên tục nên phù hợp regression.

### 2. Tại sao preprocessing?
Dữ liệu có missing values, categorical variables và outliers. Model không thể xử lý trực tiếp categorical text và dữ liệu chưa xử lý.

### 3. Tại sao train/test split?
Để đo khả năng tổng quát trên dữ liệu chưa từng dùng khi train. Preprocessing chỉ fit trên train để tránh data leakage.

### 4. Tại sao Feature Engineering?
Để tạo tín hiệu sát nghiệp vụ hơn như product popularity, customer lifetime value và seasonal trend.

### 5. Tại sao thử nhiều model?
Không giả định trước model tốt nhất; cần thực nghiệm và so sánh bằng cùng metric.

### 6. Vì sao chọn Ridge?
Tuned Ridge cho RMSE thấp nhất trong lần chạy hiện tại.

### 7. `alpha` của Ridge là gì?
Đây là mức regularization L2. GridSearchCV được dùng để chọn giá trị phù hợp thay vì chọn cảm tính.

### 8. Tại sao dùng MAE, MSE, RMSE, R²?
MAE dễ diễn giải; MSE phạt lỗi lớn; RMSE cùng đơn vị với target; R² đo mức độ biến thiên được mô hình giải thích.

### 9. Tại sao xử lý outlier?
Outlier có thể kéo lệch mô hình và làm việc scale/fit kém ổn định. Project dùng IQR clipping học từ train set.

### 10. Model được lưu ở đâu?
`modeling/best_model.pkl`; preprocessor ở `data/processed/preprocessor.pkl`; metrics ở `modeling/best_metrics.json`.

## HOW

### 11. Input đi qua pipeline như thế nào?
Scenario mới → FeatureEngineer → IQR clipping → imputation/scaling/One-Hot → Tuned Ridge → predicted sales.

### 12. Chạy demo bằng lệnh nào?
`streamlit run app_streamlit.py`.

## LIMITATIONS

Dataset hiện là synthetic. Khi chuyển sang dữ liệu thực tế cần kiểm tra temporal validation, drift, bias và business impact.
