# Chọn thuật toán - Practice 02

Các ứng viên: Linear Regression, Ridge Regression, Decision Tree Regressor, Random Forest Regressor và Gradient Boosting Regressor.

**Thuật toán được chọn: Tuned Ridge Regression.**

Ridge giữ cấu trúc tuyến tính dễ giải thích của Linear Regression nhưng thêm L2 regularization, hữu ích khi feature space có nhiều biến sau One-Hot Encoding và có tương quan giữa các biến.

Trong thực nghiệm, Tuned Ridge với `alpha=10` đạt RMSE thấp nhất trong bảng so sánh: MAE ≈ 14.393, MSE ≈ 324.924, RMSE ≈ 18.026, R² ≈ 0.725.

Kết luận được đưa ra từ thực nghiệm, không mặc định rằng Random Forest hay Gradient Boosting luôn tốt hơn.
