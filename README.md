# 🌍 Earthquake Magnitude Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting earthquake magnitude using machine learning techniques and seismic/geographical data.

The workflow includes:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Outlier Handling
- Feature Selection
- Hyperparameter Tuning
- Model Evaluation
- Explainable AI (Feature Importance & SHAP)

The final model was built using ensemble learning algorithms such as Random Forest and XGBoost.

---

LIVE LINK - https://earthquakeprediction-by-sourabh.streamlit.app/


# 🚀 Problem Statement

Earthquakes are one of the most dangerous natural disasters. Predicting earthquake magnitude can help improve:

- Disaster preparedness
- Risk management
- Seismic analysis
- Emergency response systems

This project aims to build a regression model capable of predicting earthquake magnitude using historical earthquake data.

---

# 📂 Dataset Information

The dataset contains seismic and geographical earthquake information.

## 🔹 Numerical Features

- latitude
- longitude
- depth
- nst
- gap
- rms
- horizontalError
- depthError

## 🔹 Categorical Features

- magType
- locationSource
- magSource
- country
- depth_category

## 🔹 Time-Based Features

- year
- month
- day
- hour

## 🎯 Target Variable

- mag → Earthquake Magnitude

---

# 📊 Exploratory Data Analysis (EDA)

EDA techniques used:

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

### Country Extraction

Country names were extracted from the place feature.

### Time Feature Extraction

Extracted:

- year
- month
- day
- hour

from timestamp columns.

### Depth Categorization

Earthquake depth categorized into:

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
- Gradient Boosting Regressor
- XGBoost Regressor

---

# 🔥 Final Model

## ✅ XGBoost Regressor

The final selected model was XGBoost Regressor due to:

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

The following regression metrics were used:

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

- nst
- gap
- depth
- latitude
- longitude
- country
- year

Feature importance analysis helped identify meaningful seismic patterns influencing earthquake magnitude.

---

# 🔍 Explainable AI (SHAP)

SHAP analysis was performed to:

- Explain model predictions
- Identify feature contributions
- Improve interpretability

---

# 📊 Visualizations Included

The project includes:

- Correlation Heatmap
- Residual Plot
- Actual vs Predicted Plot
- Feature Importance Graph
- SHAP Summary Plot
- Distributio



