import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = 'file.csv'
df = pd.read_csv(file_path)

# Display the first few rows to understand the structure
print(df.head())