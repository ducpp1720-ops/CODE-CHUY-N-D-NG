# Dataset Justification – Predicting Product Sales

## 1. Dataset được chọn

Project sử dụng một **synthetic e-commerce product sales dataset** được tạo bằng `data/raw/generate_dataset.py`, seed cố định `42` để tái tạo.

Dataset có **6.000 observations và 14 columns**, target `product_sales` là biến liên tục.

## 2. Đánh giá theo Relevance – Quality – Size – Features

| Tiêu chí | Đánh giá | Giải trình |
|---|---|---|
| **Relevance** | Đạt | Có product information, customer demographics, purchase history, marketing và seasonal/holiday information đúng định hướng đề. |
| **Quality** | Đạt | Dataset được sinh có kiểm soát và cố ý chứa missing/outlier để thực hành data preparation; pipeline có kiểm tra duplicate, IQR clipping và imputation. |
| **Size** | Đạt | 6.000 observations đủ cho train/test, so sánh nhiều regression models và tuning trong môi trường lab. |
| **Features** | Đạt | Có numeric, categorical và temporal variables; bổ sung `product_popularity`, `customer_lifetime_value`, `seasonal_index`, `month`, `quarter`, `day_of_week`, `ad_efficiency`. |

## 3. Mapping với đề Practice 02

- **Product:** `product_category`, `price`, `number_of_reviews`, `rating`.
- **Customer demographics:** `customer_age`, `gender`, `location`.
- **Purchase history:** `previous_purchases`, `purchase_frequency`.
- **Marketing:** `campaigns_run`, `ad_spend`.
- **Seasonal/holiday:** `purchase_date`, `is_holiday` và temporal engineered features.
- **Target:** `product_sales`.

Đề gợi ý Kaggle nhưng không bắt buộc dataset cụ thể. Project vì vậy minh bạch nguồn là **synthetic dataset generated locally**, không gọi dữ liệu mô phỏng là dữ liệu Kaggle.
