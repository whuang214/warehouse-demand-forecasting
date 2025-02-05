# Warehouse Demand Forecasting & Customer Churn Prediction

## Overview
This repository contains a data science project focusing on **warehouse demand forecasting** and **customer churn prediction**. The project utilizes **time-series forecasting models**, **logistic regression**, and **K-Nearest Neighbors (KNN) classification** to analyze demand trends and customer behavior.

## Project Structure
- **`1WarehouseForecasting.py`** – Implements multiple time-series forecasting models to predict **product demand at Warehouse J**.
- **`2LogisticModel.py`** – Develops a **logistic regression model** to predict customer churn based on demographic and service-related attributes.
- **`3KNNChurnClassification.py`** – Implements **KNN classification** for customer churn prediction, optimizing the best `k` value.
- **`WarehouseProductDemand.csv`** – Dataset containing warehouse demand data for forecasting.
- **`CustomerData_Composite-3.csv`** – Customer dataset for churn prediction analysis.

## Features & Methodologies

### 1. Warehouse Demand Forecasting (`1WarehouseForecasting.py`)
- **Time-Series Models Implemented:**
  - **Linear Regression** – Fits a simple trend to demand over time.
  - **Exponential Regression** – Captures exponential growth or decay patterns.
  - **Polynomial Regression (2nd Degree)** – Fits a quadratic trend.
  - **Seasonality-Only Model** – Uses monthly dummies to capture seasonality.
  - **Seasonality + Trend Model** – Combines trend and seasonal effects.
  - **ARIMA (1,1,1)** – AutoRegressive Integrated Moving Average for short-term forecasting.
  - **SARIMA (1,1,1)(1,1,1,12)** – Seasonal ARIMA incorporating seasonal patterns.
  - **Holt-Winters (Additive)** – Captures trend and seasonality with exponential smoothing.

### 2. Logistic Regression for Customer Churn (`2LogisticModel.py`)
- **Encodes categorical features** (e.g., contract type, online security, streaming TV).
- **Fits a logistic regression model** to predict churn probability.
- **Computes odds ratios and log-odds** to interpret feature importance.
- **Creates Decile Lift Charts & Cumulative Gains Charts** to assess model effectiveness.
- **Findings:**
  - **Senior citizens, streaming TV users, and paperless billing customers** have a higher risk of churn.
  - **Long-term contracts, online security services, and referrals** help retain customers.

### 3. K-Nearest Neighbors (KNN) for Churn Classification (`3KNNChurnClassification.py`)
- **Prepares and standardizes features** (e.g., age, satisfaction score, CLTV, churn score).
- **Optimizes `k` value** using cross-validation to minimize error rates.
- **Findings:**
  - The best-performing `k` was **8**, achieving an accuracy of **95.88%**.
  - The model effectively classifies churn vs. non-churn customers.

## Results & Insights

### Warehouse Demand Forecasting Performance
| Model                              | MAE        | RMSE       | MAPE  |
|------------------------------------|------------|------------|--------|
| Linear Regression                 | 4,031,551  | 4,781,161  | 9.15%  |
| Exponential Regression            | 3,924,461  | 4,663,132  | 8.91%  |
| Polynomial Regression (2nd Degree) | 3,577,399  | 4,115,215  | 7.75%  |
| Seasonality-Only Regression        | **3,292,137** | **3,903,240** | **7.02%** |
| Seasonality + Trend Regression     | 4,213,761  | 5,153,633  | 9.42%  |
| ARIMA (1,1,1)                      | 3,399,481  | 4,067,498  | 7.11%  |
| SARIMA (1,1,1)(1,1,1,12)           | 4,912,612  | 5,777,529  | 10.50% |
| Holt-Winters (Additive)            | 3,693,096  | 4,645,288  | 8.27%  |

- **Best Model:** `Seasonality-Only Regression` had the lowest **MAE (3,292,137)** and **MAPE (7.02%)**, indicating that demand follows strong seasonal cycles.
- **ARIMA also performed well**, reinforcing the importance of seasonal patterns in warehouse demand.

### Customer Churn Prediction Results
- **Logistic Regression Findings:**
  - **Paperless billing & streaming TV users are more likely to churn.**
  - **Long-term contracts and security services reduce churn.**
- **KNN Classification Accuracy:**
  - Best `k` = **8** with **95.88% accuracy** on the test set.
  - Model successfully distinguishes churn-prone customers from retained ones.