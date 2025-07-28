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