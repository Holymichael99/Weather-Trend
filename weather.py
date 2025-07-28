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