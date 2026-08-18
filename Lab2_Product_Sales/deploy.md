# Deployment - Streamlit

## Local run

```powershell
cd Lab2_Product_Sales
pip install -r requirements.txt
streamlit run app_streamlit.py
```

The app loads the saved preprocessor and best model, accepts a product/customer/marketing scenario, transforms it using the same feature pipeline and returns predicted product sales.

## Production considerations

- Pin dependency versions for reproducibility.
- Validate input ranges and categorical values.
- Log model version and prediction timestamp.
- Monitor data drift and prediction error after real sales become available.
- Retrain periodically with recent production data.
