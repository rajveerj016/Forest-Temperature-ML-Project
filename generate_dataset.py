# Dataset Generator for Forest Temperature & Fire Classification
# Author: Rajveer Choudhary

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n = 500

# Generate synthetic data
temperature = np.random.normal(30, 5, n)          # Average 30°C, some variance
humidity = np.random.uniform(20, 90, n)           # %
wind = np.random.uniform(0, 20, n)                # km/h
rain = np.random.uniform(0, 100, n)               # mm

# Fire-prone classification rule (synthetic logic)
# More likely fire if: high temp, low humidity, low rain
fire_prone = (
    (temperature > 32).astype(int) &
    (humidity < 40).astype(int) &
    (rain < 20).astype(int)
).astype(int)

# Create DataFrame
data = pd.DataFrame({
    "temp": temperature,
    "humidity": humidity,
    "wind": wind,
    "rain": rain,
    "fire_prone": fire_prone
})

# Save dataset
data.to_csv("dataset.csv", index=False)

print("✅ dataset.csv generated successfully!")
print(data.head())