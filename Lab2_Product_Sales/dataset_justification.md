# Dataset Justification - Predicting Product Sales

## Dataset
Synthetic e-commerce product sales dataset generated locally with a fixed seed for reproducibility. The project does not claim it is a Kaggle dataset because the Practice 02 brief does not mandate a particular dataset.

## Relevance
Contains product information, customer demographics, purchase history, marketing campaign variables, seasonal/holiday information and numerical target `product_sales`.

## Quality
The pipeline checks missing values, duplicates and outliers. Numeric variables use median imputation; categorical variables use most-frequent imputation and One-Hot Encoding. Price outliers are clipped using an IQR rule learned from the training partition.

## Size
6,000 observations. This is large enough for an instructional train/test split and comparison of several regression algorithms while remaining practical for a lab environment.

## Features
The raw schema includes product category, price, customer age, gender, location, previous purchases, purchase frequency, campaigns run, ad spend, number of reviews, rating, holiday flag and purchase date. Feature engineering adds product popularity, customer lifetime value and seasonal/time features.

## Reproducibility
Run `python data/raw/generate_dataset.py` to recreate the dataset with seed 42.
