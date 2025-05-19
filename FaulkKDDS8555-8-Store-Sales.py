import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from joblib import Parallel, delayed
import os

# Load the dataset
train = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Store_Sales/train.csv")
test = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Store_Sales/test.csv")

# Create output directories
os.makedirs("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Store_Sales/plots", exist_ok=True)

# Function to forecast a single store-family combo
def process_combo(store, family):
    print(f"Processing Store {store} - Family {family}")
    
    train_subset = train[(train['store_nbr'] == store) & (train['family'] == family)].copy()
    test_subset = test[(test['store_nbr'] == store) & (test['family'] == family)].copy()
    test_ids = test_subset[['id']].copy()

    fallback_sales = test_ids.copy()
    fallback_sales['sales'] = 0.0

    if train_subset.empty:
        return fallback_sales, fallback_sales, fallback_sales

    train_subset['date'] = pd.to_datetime(train_subset['date'])
    train_series = train_subset.set_index('date')['sales'].asfreq('D').fillna(0)
    forecast_horizon = len(test_subset)
    forecast_dates = pd.date_range(start=train_series.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')

    if train_series.nunique() <= 1:
        mean_val = train_series.mean()
        fallback_sales['sales'] = mean_val
        return fallback_sales, fallback_sales, fallback_sales

    try:
        ets_model = ExponentialSmoothing(train_series, trend='add', seasonal='add', seasonal_periods=365).fit()
        ets_forecast = ets_model.forecast(forecast_horizon)

        arima_model = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
        arima_forecast = arima_model.forecast(forecast_horizon)

        ensemble_forecast = (ets_forecast + arima_forecast) / 2

        # Save plots
        plt.figure(figsize=(10, 5))
        plt.plot(train_series, label='Train')
        plt.plot(forecast_dates, ets_forecast, label='ETS Forecast', color='red')
        plt.plot(forecast_dates, arima_forecast, label='ARIMA Forecast', color='green')
        plt.title(f"Forecast: Store {store}, Family {family}")
        plt.legend()
        plt.savefig(f"/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Store_Sales/plots/forecast_{store}_{family}.png")
        plt.close()

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(train_series, ax=axes[0])
        plot_pacf(train_series, ax=axes[1])
        axes[0].set_title(f"ACF: Store {store}, Family {family}")
        axes[1].set_title(f"PACF: Store {store}, Family {family}")
        plt.tight_layout()
        plt.savefig(f"/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Store_Sales/plots/acf_pacf_{store}_{family}.png")
        plt.close()

        temp_ets = test_ids.copy()
        temp_ets['sales'] = ets_forecast.values

        temp_arima = test_ids.copy()
        temp_arima['sales'] = arima_forecast.values

        temp_ensemble = test_ids.copy()
        temp_ensemble['sales'] = ensemble_forecast.values

        return temp_ets, temp_arima, temp_ensemble

    except Exception as e:
        print(f"Model failed for Store {store}, Family {family}: {e}")
        return fallback_sales, fallback_sales, fallback_sales

# Get all combinations
combos = [(store, family) for store in test['store_nbr'].unique() for family in test[test['store_nbr'] == store]['family'].unique()]

# Run in parallel
results = Parallel(n_jobs=-1)(delayed(process_combo)(store, family) for store, family in combos)

# Combine results
submission_ets = pd.concat([res[0] for res in results])
submission_arima = pd.concat([res[1] for res in results])
submission_ensemble = pd.concat([res[2] for res in results])

# Save CSVs
submission_ets.to_csv("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Store_Sales/submission_ets.csv", index=False)
submission_arima.to_csv("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Store_Sales/submission_arima.csv", index=False)
submission_ensemble.to_csv("/Users/kevinfaulk/Documents/DDS-8555/Assignment 8/Store_Sales/submission_ensemble.csv", index=False)
print("Submissions saved with parallel processing.")
