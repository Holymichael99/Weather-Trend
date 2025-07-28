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