import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import BaggingRegressor
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

def calculate_metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Directional_Accuracy": np.mean(np.sign(y_true - y_pred) == 1),
    }

def create_features(data, target_days=1):
    df = data.copy()
    df['Returns'] = df['Close'].pct_change()

    # Create lagged features
    for lag in [1, 5, 10, 20]:
        df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)

    # Create rolling window features (using only past data)
    for window in [5, 10, 20]:
        df[f'Returns_rolling_mean_{window}'] = df['Returns'].rolling(window=window).mean().shift(1)
        df[f'Returns_rolling_std_{window}'] = df['Returns'].rolling(window=window).std().shift(1)

    # Create price-based features
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']

    # Calculate future return (target variable)
    df['Future_Return'] = df['Close'].pct_change(periods=target_days).shift(-target_days)

    return df.dropna()

def main():
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2025-04-25'
    target_days = 20  # Predict return 20 days ahead

    # Load data
    data = pd.read_csv(f"data_cache/{ticker}_data.csv", index_col=0, parse_dates=True)
    data = data.sort_index().loc[start_date:end_date]

    # Create features
    data = create_features(data, target_days)

    # Split data (maintaining temporal order)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]

    # Prepare features and target
    features = [col for col in data.columns if col not in ['Future_Return', 'Date']]
    X_train = train_data[features]
    y_train = train_data['Future_Return']
    X_test = test_data[features]
    y_test = test_data['Future_Return']

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create stratified folds
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Hyperparameter tuning for XGBoost
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=kfold, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)

    # Create a bagging ensemble with the best XGBoost model
    best_xgb = grid_search.best_estimator_
    print("Best XGBoost parameters:", best_xgb.get_params())

    bagging_xgb = BaggingRegressor(
        estimator=best_xgb,
        n_estimators=10,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        bootstrap_features=False,
        n_jobs=-1,
        random_state=42
    )
    bagging_xgb.fit(X_train_scaled, y_train)

    # Make predictions
    train_pred = bagging_xgb.predict(X_train_scaled)
    test_pred = bagging_xgb.predict(X_test_scaled)

    # Calculate metrics
    train_metrics = calculate_metrics(y_train, train_pred)
    test_metrics = calculate_metrics(y_test, test_pred)
    print("Training Metrics:")
    print(train_metrics)
    print("\nTest Metrics:")
    print(test_metrics)

    # Plot results
    plt.figure(figsize=(14, 8))
    plt.plot(train_data.index, y_train, label='Train Actual', alpha=0.7)
    plt.plot(test_data.index, y_test, label='Test Actual', alpha=0.7)
    plt.plot(test_data.index, test_pred, label='Test Predicted', alpha=0.7)
    plt.title(f'{ticker} Future {target_days}-Day Return Forecasting with XGBoost')
    plt.xlabel('Date')
    plt.ylabel('Future Return (%)')
    plt.grid(True)
    plt.legend()

    # Add metrics to the plot
    metrics_text = (
        f"Test Metrics:\n"
        f"MSE: {test_metrics['MSE']:.6f}\n"
        f"RMSE: {test_metrics['RMSE']:.6f}\n"
        f"MAE: {test_metrics['MAE']:.6f}\n"
        f"R2: {test_metrics['R2']:.2%}\n"
        f"Dir. Acc: {test_metrics['Directional_Accuracy']:.2%}"
    )
    plt.text(0.02, 0.98, metrics_text, color="purple", transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # Print feature importances (average of all base estimators)
    importances = np.mean([estimator.feature_importances_ for estimator in bagging_xgb.estimators_], axis=0)
    feature_imp = pd.DataFrame(sorted(zip(importances, features)), columns=['Value','Feature'])
    print("\nTop 10 most important features:")
    print(feature_imp.nlargest(10, 'Value'))

if __name__ == "__main__":
    main()