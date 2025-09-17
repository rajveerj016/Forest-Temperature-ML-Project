# Forest Temperature ML Project

This project predicts forest temperature and classifies fire-prone areas using Machine Learning.

## Features
- Data cleaning and Exploratory Data Analysis (EDA)
- Random Forest Regressor for temperature prediction
- Random Forest Classifier for fire-prone classification
- Hyperparameter tuning with RandomizedSearchCV

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate dataset:
   ```bash
   python3 generate_dataset.py
   ```
3. Run the ML project:
   ```bash
   python3 forest_temperature.py
   ```

## Sample Output

When running `forest_temperature.py` with the generated dataset, you can expect outputs like:

```plaintext
A histogram plot of the temperature prediction errors will also be displayed:

- **X-axis:** Prediction Error (°C)  
- **Y-axis:** Frequency of errors  
```

This helps visualize how accurate the Random Forest model is at predicting forest temperatures.