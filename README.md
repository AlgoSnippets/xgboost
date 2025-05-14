## Stock Returns Prediction with XGBoost Ensemble

**Predict future stock returns using advanced feature engineering and an XGBoost ensemble.**

---

### **Overview**

This project forecasts future stock returns (e.g., for AAPL) using historical price and volume data. It leverages feature engineering, hyperparameter-tuned XGBoost, and bagging for robust time-series regression.

---

### **Features**

- Predicts N-day future returns (default: 20 days ahead)
- Advanced lagged and rolling window features
- Hyperparameter tuning with GridSearchCV
- Bagging ensemble for improved generalization
- Performance metrics: MSE, RMSE, MAE, R², Directional Accuracy
- Visualizes actual vs. predicted returns with metrics overlay

---

### **Usage**

1. **Prepare Data:**
   Place your stock CSV (e.g., `AAPL_data.csv`) in `data_cache/` with columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
2. **Run the Script:**

```bash
python main.py
```

3. **Output:**
   - Forecast plot saved in `x/`
   - Console output: best parameters, metrics, top features

---

### **Requirements**

- Python 3.7+
- numpy, pandas, scikit-learn, matplotlib, xgboost

---

### **Customization**

- Change `ticker`, `start_date`, `end_date`, or `target_days` in `main()`
- Adjust feature engineering in `create_features()`

---

### **Example Plot**

Forecasts and metrics are visualized for easy interpretation.

---

**Note:** This code is for educational and research purposes only. Not investment advice.
