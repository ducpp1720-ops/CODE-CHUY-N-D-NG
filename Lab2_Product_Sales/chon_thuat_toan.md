# Chọn thuật toán — Practice 02

## Các ứng viên

- **Linear Regression:** baseline tuyến tính, dễ giải thích.
- **Ridge Regression:** thêm L2 regularization, hữu ích khi có nhiều đặc trưng sau One-Hot Encoding và các biến tương quan.
- **Decision Tree Regressor:** học quan hệ phi tuyến và tương tác giữa biến.
- **Random Forest Regressor:** ensemble nhiều cây, giúp giảm variance.
- **Gradient Boosting Regressor:** xây dựng cây tuần tự để giảm residual error.

## Tiêu chí chọn

Tất cả model được train theo cùng pipeline và đánh giá bằng **MAE, MSE, RMSE và R²**. Tiêu chí chính là **RMSE thấp nhất**.

Sau model comparison, Ridge được tuning bằng `GridSearchCV` 5-fold trên `alpha = [0.01, 0.1, 1, 10, 50, 100]`.

## Kết quả hiện tại

**Tuned Ridge Regression** được chọn với `alpha = 10`.

- MAE ≈ 14.487
- MSE ≈ 332.030
- RMSE ≈ 18.222
- R² ≈ 0.719

Không chọn model theo cảm tính; kết quả được sinh tự động từ `modeling/train_models.py`.
