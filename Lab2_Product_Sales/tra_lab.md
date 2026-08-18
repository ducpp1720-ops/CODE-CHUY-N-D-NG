# Kịch bản trả Lab Practice 02

## 1. Giới thiệu
Bài toán: dự đoán doanh số sản phẩm trong bối cảnh thương mại điện tử để hỗ trợ tồn kho và marketing.

## 2. Dataset
Dataset gồm thông tin sản phẩm, khách hàng, lịch sử mua hàng, marketing và yếu tố thời gian/mùa vụ. Target là `sales` (biến số liên tục), nên đây là bài toán supervised learning - regression.

## 3. Preprocessing
- Kiểm tra missing values, duplicate và outlier.
- Numeric: median imputation + scaling.
- Categorical: imputation + One-Hot Encoding.
- Chia train/test trước khi fit preprocessing để tránh data leakage.

## 4. Feature Engineering
Tạo các biến có ý nghĩa nghiệp vụ như customer lifetime value, purchase frequency, review/popularity signal và seasonal/month features.

## 5. Model
So sánh Linear Regression, Ridge, tuned Ridge, Decision Tree, Random Forest và Gradient Boosting. Tuned Ridge được chọn theo RMSE thấp nhất trên test set.

## 6. Evaluation
Dùng MAE, MSE, RMSE và R². RMSE là metric chính vì cùng đơn vị với sales và phạt sai số lớn.

## 7. Deployment
Mô hình được lưu bằng joblib và có Streamlit app để nhập scenario mới và dự đoán sales.

## 8. Câu hỏi thường gặp
**Tại sao là regression?** Vì sales là số liên tục.

**Tại sao phải preprocessing?** Để xử lý missing/categorical/numeric scale và tạo đầu vào nhất quán cho model.

**Tại sao không chỉ dùng một model?** So sánh nhiều model giúp có baseline và chọn model dựa trên metric thay vì cảm tính.

**Tại sao chọn Tuned Ridge?** Vì trong thực nghiệm nó có RMSE thấp nhất trong bảng so sánh.

**R² = 0.725 nghĩa là gì?** Mô hình giải thích khoảng 72.5% biến thiên của sales trên tập test; phần còn lại đến từ yếu tố chưa được mô hình hóa và nhiễu.

**Deployment dùng gì?** Streamlit + joblib; mục tiêu là minh họa inference cho các scenario sản phẩm/campaign mới.
