import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = 'file.csv'
df = pd.read_csv(file_path)

# Display the first few rows to understand the structure
print(df.head())

# Check for the relevant columns
print(df.columns)

# Assuming the relevant columns are named 'temperature', 'humidity', 'wind_speed', 'pressure'
# If the column names are different, please specify or adjust accordingly.

# Calculate correlation matrix for temperature with other variables
correlation_matrix = df[['temperature', 'humidity', 'wind_speed', 'pressure']].corr()
print(correlation_matrix)