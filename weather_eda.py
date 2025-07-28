import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot


# Load the CSV file
file_path = 'file.csv'
df = pd.read_csv(file_path, encoding='ascii')

# Display the first few rows to understand the structure
print(df.head())

# Convert 'Date/Time' to datetime
df['Date/Time'] = pd.to_datetime(df['Date/Time'])

# Set 'Date/Time' as index
df.set_index('Date/Time', inplace=True)

# Plot temperature over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.index, y='Temp_C')
plt.title('Temperature Over Time')
plt.xlabel('Time')
plt.ylabel('Temperature (C)')
plt.show()

# Plot humidity over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.index, y='Rel Hum_%')
plt.title('Relative Humidity Over Time')
plt.xlabel('Time')
plt.ylabel('Relative Humidity (%)')
plt.show()

# Plot wind speed over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.index, y='Wind Speed_km/h')
plt.title('Wind Speed Over Time')
plt.xlabel('Time')
plt.ylabel('Wind Speed (km/h)')
plt.show()

# Seasonal plot: average temperature by month
df['Month'] = df.index.month
monthly_avg_temp = df.groupby('Month')['Temp_C'].mean()

plt.figure(figsize=(8, 4))
sns.barplot(x=monthly_avg_temp.index, y=monthly_avg_temp.values)
plt.title('Average Temperature by Month')
plt.xlabel('Month')
plt.ylabel('Average Temperature (C)')
plt.show()

# Seasonality detection: autocorrelation plot for temperature
autocorrelation_plot(df['Temp_C'])
plt.show()

# Save the processed dataframe for further analysis
outputs_dict = {'processed_df': df}


