"""
Forest Temperature Prediction & Fire Classification
Author: Rajveer Choudhary

This project predicts forest temperature and classifies fire-prone areas using Machine Learning.

Features:
- Data cleaning and EDA
- Random Forest Regressor for temperature prediction
- Random Forest Classifier for fire-prone classification
- Hyperparameter tuning with RandomizedSearchCV

Tech Stack:
- Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

How to Run:
1. Install requirements: pip install -r requirements.txt
2. Run: python3 forest_temperature.py
"""

# -------------------------------
# 1. Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score

# -------------------------------
# 2. Load Dataset
# -------------------------------
data = pd.read_csv("dataset.csv")  # make sure dataset.csv exists in the same folder
print("✅ Dataset Loaded Successfully")
print(data.head())

# -------------------------------
# 3. Data Cleaning
# -------------------------------
# Fill missing numeric values with column mean
data = data.fillna(data.mean(numeric_only=True))

# Clip outliers (simple approach)
for col in data.select_dtypes(include=np.number).columns:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    data[col] = np.clip(data[col], lower, upper)

print("✅ Data Cleaning Done")

# -------------------------------
# 4. Train-Test Split
# -------------------------------
# Assuming dataset has columns: ['temp', 'humidity', 'wind', 'rain', 'fire_prone']
# 'temp' -> target for regression
# 'fire_prone' -> target for classification

X = data.drop(["temp", "fire_prone"], axis=1)
y_reg = data["temp"]
y_clf = data["fire_prone"]

X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)

print("✅ Train-Test Split Done")

# -------------------------------
# 5. Random Forest Regressor (Temperature Prediction)
# -------------------------------
reg = RandomForestRegressor()

param_grid_reg = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

search_reg = RandomizedSearchCV(reg, param_distributions=param_grid_reg, n_iter=5, cv=3, random_state=42)
search_reg.fit(X_train, y_reg_train)

y_reg_pred = search_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
print(f"🌡️ Temperature Prediction RMSE: {rmse:.2f}")

# -------------------------------
# 6. Random Forest Classifier (Fire-Prone Prediction)
# -------------------------------
clf = RandomForestClassifier()

param_grid_clf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

search_clf = RandomizedSearchCV(clf, param_distributions=param_grid_clf, n_iter=5, cv=3, random_state=42)
search_clf.fit(X_train, y_clf_train)

y_clf_pred = search_clf.predict(X_test)
acc = accuracy_score(y_clf_test, y_clf_pred)
recall = recall_score(y_clf_test, y_clf_pred)

print(f"🔥 Fire Classification Accuracy: {acc:.2f}")
print(f"🔥 Fire Classification Recall: {recall:.2f}")

# -------------------------------
# 7. Visualization
# -------------------------------
plt.figure(figsize=(6,4))
sns.histplot(y_reg_test - y_reg_pred, bins=20, kde=True)
plt.title("Error Distribution - Temperature Prediction")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.show()

print("✅ Project Completed")