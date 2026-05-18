# 🌍 Earthquake Magnitude Prediction Using Machine Learning

## 📌 Project Overview

I built this project because earthquake prediction is one of those problems where machine learning can actually matter — not just academically, but in terms of real impact. Nepal sits on one of the most seismically active zones in the world. The 2015 Gorkha earthquake alone killed nearly 9,000 people. I wanted to understand what patterns exist in historical seismic data and whether a model could learn to estimate magnitude from measurable parameters.
So I took earthquake data from 1990 to 2026, cleaned it properly, engineered features from it, tested multiple models, and ended up with a tuned XGBoost regressor as the final model — deployed as a live Streamlit app.

Live App 🔗 https://earthquakeprediction-by-sourabh.streamlit.app/


What the Model Predicts :- 

Given seismic station readings and geographic inputs, the model predicts the Richter scale magnitude of an earthquake. It's a regression task — not classification. The output is a continuous magnitude value which the app then maps to a severity label (Minor / Light / Moderate / Strong / Major).

This project focuses on predicting earthquake magnitude using machine learning techniques and seismic/geographical data.

## The Journey — What Actually Happened

Data Cleaning First
The raw dataset had a lot of issues:

id, type, status, updated, magError, magNst — dropped (no predictive value or potential leakage)
time column — converted to datetime and extracted year, month, day, hour
place column — parsed to extract country name from the end of messy text strings
dmin — dropped due to too many missing values
Missing values in nst, depthError filled with median (skewed features)
Missing values in gap, rms, horizontalError filled with mean
country values cleaned — fixed 2025 Southern Tibetan Plateau Earthquake → Southern Tibetan Plateau, unified inconsistent spellings
Added depth_category feature: Shallow (0–70km) / Intermediate (70–300km) / Deep (300–700km)

---

# 📂 Dataset Information

The dataset contains seismic and geographical earthquake information.
Sources Kaggle - https://www.kaggle.com/datasets/amansinghnp/nepal-earthquake-seismicity-dataset-1990-2026


## 🔹 Numerical Features

- latitude          - longitude
- depth             - nst
- gap               - rms
- horizontalError   - depthError

## 🔹 Categorical Features

- magType     - locationSource
- magSource   - country
- depth_category

## 🔹 Time-Based Features

- year  - month
- day   - hour

## 🎯 Target Variable

- mag → Earthquake Magnitude

---

# 📊 Exploratory Data Analysis (EDA)

EDA techniques used:-

- Missing value analysis
- Duplicate row checking
- Outlier detection
- Correlation heatmap
- Target distribution analysis
- Skewness analysis
- Boxplots and histograms

---

# ⚙️ Data Preprocessing

The following preprocessing techniques were applied:

## ✅ Feature Engineering

Country names were extracted from the place feature.

### Time Feature Extraction

- year   - month
- day    - hour
from timestamp columns.

### Depth Categorization

Earthquake depth categorized into:-
- Shallow
- Intermediate
- Deep

---

## ✅ Encoding

Categorical variables were encoded using:

- One-Hot Encoding

---

## ✅ Outlier Handling

Outliers were analyzed using:

- IQR Method
- Boxplots

---

# 🤖 Machine Learning Models Used

The following models were explored:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

---

# 🔥 Final Model

## ✅ XGBoost Regressor

- Better generalization
- Strong nonlinear learning capability
- Robust performance on structured data
- Stable test performance

---

# 🎯 Hyperparameter Tuning

Hyperparameter tuning was performed using:

- RandomizedSearchCV

## Tuned Parameters

### Random Forest

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features

### XGBoost

- learning_rate
- max_depth
- n_estimators
- subsample
- colsample_bytree

---

# 📈 Model Evaluation Metrics

- R² Score
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

---

# 🏆 Final Model Performance

## XGBoost Performance

| Metric | Train | Test |
|---|---|---|
| R² Score | 0.7206 | 0.6154 |
| MAE | 0.1719 | 0.2012 |
| RMSE | 0.2277 | 0.2629 |

---

# 📌 Feature Importance Analysis

Important features identified:

- nst         - gap
- depth       - latitude
- longitude   - country
- year

Feature importance analysis helped identify meaningful seismic patterns influencing earthquake magnitude.

---

# 📊 Visualizations Included

The project includes:

- Correlation Heatmap
- Residual Plot
- Actual vs Predicted Plot
- Feature Importance Graph
- Distribution

Tech Stack

Python 3
Pandas, NumPy — data cleaning, feature engineering
Matplotlib, Seaborn — EDA visualizations
Scikit-learn — multiple models, train-test split, RandomizedSearchCV
XGBoost — final model
Pickle — model serialization
Streamlit — live web app


What I Learned
The most interesting part of this project was feature leakage. magType and magSource columns seem like useful features — they describe how the magnitude was measured. But they're assigned after the earthquake is recorded, not before. Including them would give the model information it couldn't possibly have at prediction time. Dropping them was the right call, even though it reduced performance slightly.
The second lesson was that depth matters more than I expected. Shallow earthquakes (under 70km) behave very differently from deep ones. Creating depth_category as an explicit feature helped the model learn this boundary more cleanly than raw depth alone.

Author
Sourabh Vishwakarma
LinkedIn - www.linkedin.com/in/sourabh9098
GitHub - https://github.com/sourabh9098/




