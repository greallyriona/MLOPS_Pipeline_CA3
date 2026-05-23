import pandas as pd

# Load data
data = pd.read_csv("data/data.csv")

# Clean column names
data.columns = data.columns.str.strip()

# Save cleaned version
data.to_csv("data/cleaned_data.csv", index=False)

print("Data cleaned")