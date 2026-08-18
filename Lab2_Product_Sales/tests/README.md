# Tests

Run the lightweight reproducibility check with:

```powershell
python tests/smoke_test.py
```

The test loads the saved preprocessing pipeline and model, transforms one new scenario, predicts sales, and checks the selected model/alpha.