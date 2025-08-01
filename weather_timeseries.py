import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the CSV file
file_path = 'file.csv'
df = pd.read_csv(file_path, encoding='ascii')

# Resample data to daily frequency for smoother trends
daily_df = df.resample('D').mean()

# Prepare data for Prophet
prophet_df = daily_df.reset_index()
prophet_df.columns = ['ds', 'y']# Forecasting function

def forecast_time_series(series, model_type='ARIMA', steps=7):
    if model_type == 'ARIMA':
        model = SARIMAX(series, order=(1,1,1))
        result = model.fit(disp=False)
        forecast = result.get_forecast(steps=steps)
        pred_ci = forecast.conf_int()
        return forecast.predicted_mean, pred_ci
    elif model_type == 'Prophet':
        m = Prophet()
        df_prophet = series.reset_index()
        df_prophet.columns = ['ds', 'y']
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=steps)
        forecast = m.predict(future)
        return forecast[['ds', 'yhat']].tail(steps)
    else:
        raise ValueError('Unsupported model type')
    
    # Forecast Temperature
temp_series = daily_df['Temp_C'].dropna()
temp_forecast, temp_ci = forecast_time_series(temp_series, 'ARIMA')

# Forecast Humidity
hum_series = daily_df['Rel Hum_%'].dropna()
hum_forecast, hum_ci = forecast_time_series(hum_series, 'ARIMA')

# Forecast Pressure
press_series = daily_df['Press_kPa'].dropna()
press_forecast, press_ci = forecast_time_series(press_series, 'ARIMA')

# Plotting the forecasts
plt.figure(figsize=(15, 10))

# Temperature
plt.subplot(3, 1, 1)
plt.plot(temp_series.index, temp_series, label='Actual Temp')
plt.plot(temp_series.index[-1] + pd.to_timedelta(range(1,8), 'D'), temp_forecast, label='Forecast Temp')
plt.fill_between(temp_series.index[-1] + pd.to_timedelta(range(1,8), 'D'), temp_ci.iloc[:, 0], temp_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Temperature Forecast')
plt.legend()

# Humidity
plt.subplot(3, 1, 2)
plt.plot(hum_series.index, hum_series, label='Actual Humidity')
plt.plot(hum_series.index[-1] + pd.to_timedelta(range(1,8), 'D'), hum_forecast, label='Forecast Humidity')
plt.fill_between(hum_series.index[-1] + pd.to_timedelta(range(1,8), 'D'), hum_ci.iloc[:, 0], hum_ci.iloc[:, 1], color='lightblue', alpha=0.3)
plt.title('Humidity Forecast')
plt.legend()

# Pressure
plt.subplot(3, 1, 3)
plt.plot(press_series.index, press_series, label='Actual Pressure')
plt.plot(press_series.index[-1] + pd.to_timedelta(range(1,8), 'D'), press_forecast, label='Forecast Pressure')
plt.fill_between(press_series.index[-1] + pd.to_timedelta(range(1,8), 'D'), press_ci.iloc[:, 0], press_ci.iloc[:, 1], color='lightgreen', alpha=0.3)
plt.title('Pressure Forecast')
plt.legend()

plt.tight_layout()
plt.show()

# Prophet for comparison
print("\nProphet Temperature Forecast (next 7 days):")
prophet_forecast = forecast_time_series(temp_series, model_type='Prophet', steps=steps)
print(prophet_forecast)
