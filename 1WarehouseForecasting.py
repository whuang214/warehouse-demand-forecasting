import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import warnings

warnings.filterwarnings("ignore")


def forecast_accuracy_metrics(y_true, y_pred):
    """
    Calculate forecast accuracy metrics: MAE, RMSE, and MAPE.
    Returns a dict of { 'MAE': ..., 'RMSE': ..., 'MAPE': ... }
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    # Handle potential division by zero in MAPE
    mask = y_true != 0
    if mask.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def run_linear_regression(train_data, test_data, test_index, full_data, future_steps=6):
    """
    1) Fit Linear Regression on train_data: Demand ~ t
    2) Forecast test set
    3) Print metrics and plot test forecast with trend line across entire train + test data
    4) Fit on full_data and forecast future_steps
    5) Plot entire timeline with trend line and future forecast
    """
    # ============================
    # Forecast Test Set
    # ============================
    df_reg = train_data.reset_index().copy()
    df_reg["t"] = np.arange(1, len(df_reg) + 1)

    # Fit linear model
    model = smf.ols("Demand ~ t", data=df_reg).fit()

    # Prepare future for test set
    future_t_test = np.arange(len(df_reg) + 1, len(df_reg) + 1 + len(test_data))
    df_future_test = pd.DataFrame({"t": future_t_test})

    # Forecast for test set
    forecast_vals_test = model.predict(df_future_test)
    forecast_series_test = pd.Series(forecast_vals_test.values, index=test_index)

    # Compute metrics for test set
    metrics_test = forecast_accuracy_metrics(
        test_data.values, forecast_series_test.values
    )

    print("=== Linear Regression ===")
    print("Forecasting Test Set:")
    print(f"MAE:  {metrics_test['MAE']:.2f}")
    print(f"RMSE: {metrics_test['RMSE']:.2f}")
    print(f"MAPE: {metrics_test['MAPE']:.2f}%\n")

    # ============================
    # Plot Test Set Forecast with Trend Line
    # ============================
    # Predict on train data for trend line
    fitted_train = model.predict(df_reg)

    # Predict on test data for trend line
    fitted_test = model.predict(df_future_test)

    # Combine train and test fitted values
    fitted_full = pd.concat(
        [
            pd.Series(fitted_train.values, index=train_data.index),
            pd.Series(fitted_test.values, index=test_data.index),
        ]
    )

    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data.values, label="Train", color="blue")
    plt.plot(test_data.index, test_data.values, label="Test", color="orange")
    plt.plot(
        fitted_full.index,
        fitted_full.values,
        label="Fitted Trend",
        color="red",
        linestyle="--",
    )
    plt.title("Linear Regression Forecast - Test Set")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ============================
    # Forecast Future Set
    # ============================
    # Refit model on full data
    df_full = full_data.reset_index().copy()
    df_full["t"] = np.arange(1, len(df_full) + 1)
    model_full = smf.ols("Demand ~ t", data=df_full).fit()

    # Prepare future for forecasting beyond dataset
    future_t_future = np.arange(len(df_full) + 1, len(df_full) + 1 + future_steps)
    df_future_future = pd.DataFrame({"t": future_t_future})

    # Forecast future
    forecast_vals_future = model_full.predict(df_future_future)
    last_date = full_data.index[-1]
    future_dates = pd.date_range(
        last_date + pd.offsets.MonthEnd(1), periods=future_steps, freq="M"
    )
    forecast_series_future = pd.Series(forecast_vals_future.values, index=future_dates)

    # ============================
    # Plot Future Forecast with Trend Line
    # ============================
    # Predict on full data for trend line
    fitted_full_data = model_full.predict(df_full)

    plt.figure(figsize=(12, 6))
    plt.plot(full_data.index, full_data.values, label="Historical Demand", color="blue")
    plt.plot(
        full_data.index,
        fitted_full_data.values,
        label="Fitted Trend",
        color="red",
        linestyle="--",
    )
    plt.plot(
        forecast_series_future.index,
        forecast_series_future.values,
        label="Forecast (Linear - Future)",
        color="green",
        marker="o",
    )
    plt.title("Linear Regression Forecast - 6 Months Future")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model_full, metrics_test, forecast_series_future


def run_exponential_regression(
    train_data, test_data, test_index, full_data, future_steps=6
):
    """
    Fit Exponential Regression: log(Demand) ~ t
    Forecast test set and future_steps
    Plot test and future forecasts with trend lines across entire dataset
    """
    # ============================
    # Forecast Test Set
    # ============================
    # Filter out non-positive demands
    train_positive = train_data[train_data > 0].copy()
    if len(train_positive) < 2:
        print("=== Exponential Regression ===")
        print("Not enough positive demand data to fit exponential regression.\n")
        return None, {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}, None

    df_reg = train_positive.reset_index().copy()
    df_reg["t"] = np.arange(1, len(df_reg) + 1)
    df_reg["log_demand"] = np.log(df_reg["Demand"])

    # Fit model
    model = smf.ols("log_demand ~ t", data=df_reg).fit()

    # Prepare future for test set
    future_t_test = np.arange(len(df_reg) + 1, len(df_reg) + 1 + len(test_data))
    df_future_test = pd.DataFrame({"t": future_t_test})

    # Forecast for test set
    pred_log_test = model.predict(df_future_test)
    pred_exp_test = np.exp(pred_log_test)
    forecast_series_test = pd.Series(pred_exp_test.values, index=test_index)

    # Compute metrics for test set
    metrics_test = forecast_accuracy_metrics(
        test_data.values, forecast_series_test.values
    )

    print("=== Exponential Regression ===")
    print("Forecasting Test Set:")
    print(f"MAE:  {metrics_test['MAE']:.2f}")
    print(f"RMSE: {metrics_test['RMSE']:.2f}")
    print(f"MAPE: {metrics_test['MAPE']:.2f}%\n")

    # ============================
    # Plot Test Set Forecast with Trend Line
    # ============================
    # Predict on train data for trend line
    fitted_train = np.exp(model.predict(df_reg))

    # Predict on test data for trend line
    fitted_test = np.exp(model.predict(df_future_test))

    # Combine train and test fitted values
    fitted_full = pd.concat(
        [
            pd.Series(fitted_train.values, index=train_positive.index),
            pd.Series(fitted_test.values, index=test_data.index),
        ]
    )

    plt.figure(figsize=(12, 6))
    plt.plot(train_positive.index, train_positive.values, label="Train", color="blue")
    plt.plot(test_data.index, test_data.values, label="Test", color="orange")
    plt.plot(
        fitted_full.index,
        fitted_full.values,
        label="Fitted Trend",
        color="red",
        linestyle="--",
    )
    plt.title("Exponential Regression Forecast - Test Set")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ============================
    # Forecast Future Set
    # ============================
    # Refit model on full data
    full_positive = full_data[full_data > 0].copy()
    if len(full_positive) < 2:
        print(
            "Exponential Regression cannot be applied to full data due to insufficient positive demands."
        )
        forecast_series_future = None
        return model, metrics_test, forecast_series_future

    df_full = full_positive.reset_index().copy()
    df_full["t"] = np.arange(1, len(df_full) + 1)
    df_full["log_demand"] = np.log(df_full["Demand"])

    model_full = smf.ols("log_demand ~ t", data=df_full).fit()

    # Prepare future for forecasting beyond dataset
    future_t_future = np.arange(len(df_full) + 1, len(df_full) + 1 + future_steps)
    df_future_future = pd.DataFrame({"t": future_t_future})

    # Forecast future
    pred_log_future = model_full.predict(df_future_future)
    pred_exp_future = np.exp(pred_log_future)
    last_date = full_data.index[-1]
    future_dates = pd.date_range(
        last_date + pd.offsets.MonthEnd(1), periods=future_steps, freq="M"
    )
    forecast_series_future = pd.Series(pred_exp_future.values, index=future_dates)

    # ============================
    # Plot Future Forecast with Trend Line
    # ============================
    # Predict on full data for trend line
    fitted_full_data = np.exp(model_full.predict(df_full))

    plt.figure(figsize=(12, 6))
    plt.plot(
        full_positive.index,
        full_positive.values,
        label="Historical Demand",
        color="blue",
    )
    plt.plot(
        full_positive.index,
        fitted_full_data.values,
        label="Fitted Trend",
        color="red",
        linestyle="--",
    )
    plt.plot(
        forecast_series_future.index,
        forecast_series_future.values,
        label="Forecast (Exponential - Future)",
        color="green",
        marker="o",
    )
    plt.title("Exponential Regression Forecast - 6 Months Future")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model_full, metrics_test, forecast_series_future


def run_polynomial_regression(
    train_data, test_data, test_index, full_data, future_steps=6
):
    """
    Fit a 2nd-degree polynomial: Demand ~ t + t^2
    Forecast test set and future_steps
    Plot test and future forecasts with trend lines across entire dataset
    """
    # ============================
    # Forecast Test Set
    # ============================
    df_reg = train_data.reset_index().copy()
    df_reg["t"] = np.arange(1, len(df_reg) + 1)
    df_reg["t2"] = df_reg["t"] ** 2

    # Fit model
    model = smf.ols("Demand ~ t + t2", data=df_reg).fit()

    # Prepare future for test set
    future_t_test = np.arange(len(df_reg) + 1, len(df_reg) + 1 + len(test_data))
    df_future_test = pd.DataFrame({"t": future_t_test})
    df_future_test["t2"] = df_future_test["t"] ** 2

    # Forecast for test set
    forecast_vals_test = model.predict(df_future_test)
    forecast_series_test = pd.Series(forecast_vals_test.values, index=test_index)

    # Compute metrics for test set
    metrics_test = forecast_accuracy_metrics(
        test_data.values, forecast_series_test.values
    )

    print("=== Polynomial Regression (2nd Degree) ===")
    print("Forecasting Test Set:")
    print(f"MAE:  {metrics_test['MAE']:.2f}")
    print(f"RMSE: {metrics_test['RMSE']:.2f}")
    print(f"MAPE: {metrics_test['MAPE']:.2f}%\n")

    # ============================
    # Plot Test Set Forecast with Trend Line
    # ============================
    # Predict on train data for trend line
    fitted_train = model.predict(df_reg)

    # Predict on test data for trend line
    fitted_test = model.predict(df_future_test)

    # Combine train and test fitted values
    fitted_full = pd.concat(
        [
            pd.Series(fitted_train.values, index=train_data.index),
            pd.Series(fitted_test.values, index=test_data.index),
        ]
    )

    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data.values, label="Train", color="blue")
    plt.plot(test_data.index, test_data.values, label="Test", color="orange")
    plt.plot(
        fitted_full.index,
        fitted_full.values,
        label="Fitted Trend",
        color="red",
        linestyle="--",
    )
    plt.title("Polynomial Regression Forecast - Test Set")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ============================
    # Forecast Future Set
    # ============================
    # Refit model on full data
    df_full = full_data.reset_index().copy()
    df_full["t"] = np.arange(1, len(df_full) + 1)
    df_full["t2"] = df_full["t"] ** 2
    model_full = smf.ols("Demand ~ t + t2", data=df_full).fit()

    # Prepare future for forecasting beyond dataset
    future_t_future = np.arange(len(df_full) + 1, len(df_full) + 1 + future_steps)
    df_future_future = pd.DataFrame({"t": future_t_future})
    df_future_future["t2"] = df_future_future["t"] ** 2

    # Forecast future
    forecast_vals_future = model_full.predict(df_future_future)
    last_date = full_data.index[-1]
    future_dates = pd.date_range(
        last_date + pd.offsets.MonthEnd(1), periods=future_steps, freq="M"
    )
    forecast_series_future = pd.Series(forecast_vals_future.values, index=future_dates)

    # ============================
    # Plot Future Forecast with Trend Line
    # ============================
    # Predict on full data for trend line
    fitted_full_data = model_full.predict(df_full)

    plt.figure(figsize=(12, 6))
    plt.plot(full_data.index, full_data.values, label="Historical Demand", color="blue")
    plt.plot(
        full_data.index,
        fitted_full_data.values,
        label="Fitted Trend",
        color="red",
        linestyle="--",
    )
    plt.plot(
        forecast_series_future.index,
        forecast_series_future.values,
        label="Forecast (Polynomial - Future)",
        color="green",
        marker="o",
    )
    plt.title("Polynomial Regression Forecast - 6 Months Future")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model_full, metrics_test, forecast_series_future


def run_seasonality_only_regression(
    train_data, test_data, test_index, full_data, future_steps=6
):
    """
    Creates monthly dummies without a trend: Demand ~ month_dummies
    Forecasts test set and future_steps.
    Plots test and future forecasts with trend lines across entire dataset
    """
    # ============================
    # Forecast Test Set
    # ============================
    df_reg = train_data.reset_index().copy()
    df_reg["month"] = df_reg["Date"].dt.month.astype("category")
    df_reg_dummies = pd.get_dummies(df_reg, columns=["month"], drop_first=True)

    # Build formula: Demand ~ month_2 + month_3 + ...
    month_cols = [c for c in df_reg_dummies.columns if c.startswith("month_")]
    formula = "Demand ~ " + " + ".join(month_cols)

    # Fit model
    model = smf.ols(formula, data=df_reg_dummies).fit()

    # Prepare future for test set
    future_df_test = pd.DataFrame(
        {
            "Date": test_index,
            "month": test_index.month.astype("category"),
        }
    )
    future_df_test_dummies = pd.get_dummies(
        future_df_test, columns=["month"], drop_first=True
    )

    # Ensure all month columns exist
    for col in month_cols:
        if col not in future_df_test_dummies:
            future_df_test_dummies[col] = 0

    # Align columns
    used_cols = df_reg_dummies.columns.drop(["Date", "Demand"])
    future_df_test_dummies = future_df_test_dummies.reindex(
        columns=used_cols, fill_value=0
    )

    # Forecast for test set
    forecast_vals_test = model.predict(future_df_test_dummies)
    forecast_series_test = pd.Series(forecast_vals_test.values, index=test_index)

    # Compute metrics for test set
    metrics_test = forecast_accuracy_metrics(
        test_data.values, forecast_series_test.values
    )

    print("=== Seasonality Only Regression ===")
    print("Forecasting Test Set:")
    print(f"MAE:  {metrics_test['MAE']:.2f}")
    print(f"RMSE: {metrics_test['RMSE']:.2f}")
    print(f"MAPE: {metrics_test['MAPE']:.2f}%\n")

    # ============================
    # Plot Test Set Forecast with Seasonality Only
    # ============================
    # Predict on train data for trend line
    fitted_train = model.predict(df_reg_dummies)

    # Predict on test data for trend line
    fitted_test = model.predict(future_df_test_dummies)

    # Combine train and test fitted values
    fitted_full = pd.concat(
        [
            pd.Series(fitted_train.values, index=train_data.index),
            pd.Series(fitted_test.values, index=test_data.index),
        ]
    )

    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data.values, label="Train", color="blue")
    plt.plot(test_data.index, test_data.values, label="Test", color="orange")
    plt.plot(
        fitted_full.index,
        fitted_full.values,
        label="Fitted Seasonality",
        color="red",
        linestyle="--",
    )
    plt.title("Seasonality Only Regression Forecast - Test Set")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ============================
    # Forecast Future Set
    # ============================
    # Refit model on full data
    df_full = full_data.reset_index().copy()
    df_full["month"] = df_full["Date"].dt.month.astype("category")
    df_full_dummies = pd.get_dummies(df_full, columns=["month"], drop_first=True)

    # Ensure all month columns exist
    for col in month_cols:
        if col not in df_full_dummies:
            df_full_dummies[col] = 0

    # Align columns
    future_df_future = pd.DataFrame(
        {
            "Date": pd.date_range(
                full_data.index[-1] + pd.offsets.MonthEnd(1),
                periods=future_steps,
                freq="M",
            ),
            "month": pd.date_range(
                full_data.index[-1] + pd.offsets.MonthEnd(1),
                periods=future_steps,
                freq="M",
            ).month.astype("category"),
        }
    )
    future_df_future_dummies = pd.get_dummies(
        future_df_future, columns=["month"], drop_first=True
    )

    # Ensure all month columns exist
    for col in month_cols:
        if col not in future_df_future_dummies:
            future_df_future_dummies[col] = 0

    # Align columns
    future_df_future_dummies = future_df_future_dummies.reindex(
        columns=used_cols, fill_value=0
    )

    # Forecast future
    forecast_vals_future = model.predict(future_df_future_dummies)
    forecast_series_future = pd.Series(
        forecast_vals_future.values, index=future_df_future["Date"]
    )

    # ============================
    # Plot Future Forecast with Seasonality Only
    # ============================
    # Predict on full data for trend line
    fitted_full_data = model.predict(df_full_dummies)

    plt.figure(figsize=(12, 6))
    plt.plot(full_data.index, full_data.values, label="Historical Demand", color="blue")
    plt.plot(
        full_data.index,
        fitted_full_data.values,
        label="Fitted Seasonality",
        color="red",
        linestyle="--",
    )
    plt.plot(
        forecast_series_future.index,
        forecast_series_future.values,
        label="Forecast (Seasonality Only - Future)",
        color="green",
        marker="o",
    )
    plt.title("Seasonality Only Regression Forecast - 6 Months Future")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, metrics_test, forecast_series_future


def run_seasonal_regression(
    train_data, test_data, test_index, full_data, future_steps=6
):
    """
    Creates monthly dummies + linear trend: Demand ~ t + month_dummies
    Forecasts test set and future_steps.
    Plots test and future forecasts with trend lines across entire dataset
    """
    # ============================
    # Forecast Test Set
    # ============================
    df_reg = train_data.reset_index().copy()
    df_reg["month"] = df_reg["Date"].dt.month.astype("category")
    df_reg["t"] = np.arange(1, len(df_reg) + 1)
    df_reg_dummies = pd.get_dummies(df_reg, columns=["month"], drop_first=True)

    # Build formula: Demand ~ t + month_2 + month_3 + ...
    month_cols = [c for c in df_reg_dummies.columns if c.startswith("month_")]
    formula = "Demand ~ t + " + " + ".join(month_cols)

    # Fit model
    model = smf.ols(formula, data=df_reg_dummies).fit()

    # Prepare future for test set
    future_df_test = pd.DataFrame(
        {
            "Date": test_index,
            "t": np.arange(len(df_reg) + 1, len(df_reg) + 1 + len(test_data)),
        }
    )
    future_df_test["month"] = future_df_test["Date"].dt.month.astype("category")
    future_df_test_dummies = pd.get_dummies(
        future_df_test, columns=["month"], drop_first=True
    )

    # Ensure all month columns exist
    for col in month_cols:
        if col not in future_df_test_dummies:
            future_df_test_dummies[col] = 0

    # Align columns
    used_cols = df_reg_dummies.columns.drop(["Date", "Demand"])
    future_df_test_dummies = future_df_test_dummies.reindex(
        columns=used_cols, fill_value=0
    )

    # Forecast for test set
    forecast_vals_test = model.predict(future_df_test_dummies)
    forecast_series_test = pd.Series(forecast_vals_test.values, index=test_index)

    # Compute metrics for test set
    metrics_test = forecast_accuracy_metrics(
        test_data.values, forecast_series_test.values
    )

    print("=== Seasonal Regression (Monthly Dummies) ===")
    print("Forecasting Test Set:")
    print(f"MAE:  {metrics_test['MAE']:.2f}")
    print(f"RMSE: {metrics_test['RMSE']:.2f}")
    print(f"MAPE: {metrics_test['MAPE']:.2f}%\n")

    # ============================
    # Plot Test Set Forecast with Trend Line
    # ============================
    # Predict on train data for trend line
    fitted_train = model.predict(df_reg_dummies)

    # Predict on test data for trend line
    fitted_test = model.predict(future_df_test_dummies)

    # Combine train and test fitted values
    fitted_full = pd.concat(
        [
            pd.Series(fitted_train.values, index=train_data.index),
            pd.Series(fitted_test.values, index=test_data.index),
        ]
    )

    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data.values, label="Train", color="blue")
    plt.plot(test_data.index, test_data.values, label="Test", color="orange")
    plt.plot(
        fitted_full.index,
        fitted_full.values,
        label="Fitted Trend",
        color="red",
        linestyle="--",
    )
    plt.title("Seasonal Regression (Monthly Dummies) Forecast - Test Set")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ============================
    # Forecast Future Set
    # ============================
    # Refit model on full data
    df_full = full_data.reset_index().copy()
    df_full["month"] = df_full["Date"].dt.month.astype("category")
    df_full["t"] = np.arange(1, len(df_full) + 1)
    df_full_dummies = pd.get_dummies(df_full, columns=["month"], drop_first=True)

    # Ensure all month columns exist
    for col in month_cols:
        if col not in df_full_dummies:
            df_full_dummies[col] = 0

    # Align columns
    future_df_future = pd.DataFrame(
        {
            "Date": pd.date_range(
                full_data.index[-1] + pd.offsets.MonthEnd(1),
                periods=future_steps,
                freq="M",
            ),
            "t": np.arange(len(df_full) + 1, len(df_full) + 1 + future_steps),
        }
    )
    future_df_future["month"] = future_df_future["Date"].dt.month.astype("category")
    future_df_future_dummies = pd.get_dummies(
        future_df_future, columns=["month"], drop_first=True
    )

    # Ensure all month columns exist
    for col in month_cols:
        if col not in future_df_future_dummies:
            future_df_future_dummies[col] = 0

    # Align columns
    used_cols_seasonal = df_reg_dummies.columns.drop(["Date", "Demand"])
    future_df_future_dummies = future_df_future_dummies.reindex(
        columns=used_cols_seasonal, fill_value=0
    )

    # Forecast future
    forecast_vals_future = model.predict(future_df_future_dummies)
    forecast_series_future = pd.Series(
        forecast_vals_future.values, index=future_df_future["Date"]
    )  # Corrected Index

    # ============================
    # Plot Future Forecast with Trend Line
    # ============================
    # Predict on full data for trend line
    fitted_full_data = model.predict(df_full_dummies)

    plt.figure(figsize=(12, 6))
    plt.plot(full_data.index, full_data.values, label="Historical Demand", color="blue")
    plt.plot(
        full_data.index,
        fitted_full_data.values,
        label="Fitted Trend",
        color="red",
        linestyle="--",
    )
    plt.plot(
        forecast_series_future.index,
        forecast_series_future.values,
        label="Forecast (Seasonal Regression - Future)",
        color="green",
        marker="o",
    )
    plt.title("Seasonal Regression (Monthly Dummies) Forecast - 6 Months Future")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, metrics_test, forecast_series_future


def run_sarima(train_data, test_data, test_index, full_data, future_steps=6):
    """
    Fit a SARIMA(1,1,1)(1,1,1,12) model, forecast test set and future_steps.
    Plot test and future forecasts with trend lines across entire dataset.
    """
    # ============================
    # Forecast Test Set
    # ============================
    try:
        # Fit SARIMA model on training data
        model = SARIMAX(
            train_data,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit(disp=False)

        # Get in-sample fitted values for training data
        fitted_train = results.fittedvalues
        # drop first entry
        fitted_train = results.fittedvalues.iloc[1:]

        # Forecast for the test set
        forecast_vals_test = results.forecast(steps=len(test_data))
        forecast_series_test = pd.Series(forecast_vals_test.values, index=test_index)

        # Combine fitted_train and forecast_series_test for a continuous trend line
        fitted_full_test = pd.concat([fitted_train, forecast_series_test])

        # Compute accuracy metrics for the test set
        metrics_test = forecast_accuracy_metrics(
            test_data.values, forecast_series_test.values
        )

        print("=== SARIMA (1,1,1)(1,1,1,12) ===")
        print("Forecasting Test Set:")
        print(f"MAE:  {metrics_test['MAE']:.2f}")
        print(f"RMSE: {metrics_test['RMSE']:.2f}")
        print(f"MAPE: {metrics_test['MAPE']:.2f}%\n")
    except Exception as e:
        print("=== SARIMA (1,1,1)(1,1,1,12) ===")
        print("SARIMA failed during test set forecasting:", e, "\n")
        forecast_series_test = pd.Series([np.nan] * len(test_data), index=test_index)
        metrics_test = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
        fitted_full_test = pd.Series(
            [np.nan] * len(train_data) + [np.nan] * len(test_data),
            index=train_data.index.append(test_data.index),
        )

    # ============================
    # Plot Test Set Forecast with Trend Line
    # ============================
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data.values, label="Train", color="blue")
    plt.plot(test_data.index, test_data.values, label="Test", color="orange")
    plt.plot(
        fitted_full_test.index,
        fitted_full_test.values,
        label="Fitted Trend",
        color="red",
        linestyle="--",
    )
    plt.title("SARIMA Forecast - Test Set")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ============================
    # Forecast Future Set
    # ============================
    try:
        # Refit SARIMA model on the entire dataset (train + test)
        model_full = SARIMAX(
            full_data,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results_full = model_full.fit(disp=False)

        # Get fitted values for the entire historical data
        fitted_full_data = results_full.fittedvalues

        # Forecast future_steps ahead
        forecast_vals_future = results_full.forecast(steps=future_steps)
        future_dates = pd.date_range(
            full_data.index[-1] + pd.offsets.MonthEnd(1), periods=future_steps, freq="M"
        )
        forecast_series_future = pd.Series(
            forecast_vals_future.values, index=future_dates
        )
    except Exception as e:
        print("SARIMA failed during future forecasting:", e)
        forecast_series_future = pd.Series(
            [np.nan] * future_steps,
            index=pd.date_range(
                full_data.index[-1] + pd.offsets.MonthEnd(1),
                periods=future_steps,
                freq="M",
            ),
        )
        fitted_full_data = pd.Series([np.nan] * len(full_data), index=full_data.index)

    # ============================
    # Plot Future Forecast with Trend Line
    # ============================
    plt.figure(figsize=(12, 6))
    plt.plot(full_data.index, full_data.values, label="Historical Demand", color="blue")

    # Plot fitted trend line over historical data
    plt.plot(
        fitted_full_test.index[fitted_full_test.index >= train_data.index[0]],
        fitted_full_test[fitted_full_test.index >= train_data.index[0]],
        label="Fitted Trend",
        color="red",
        linestyle="--",
    )

    # Plot future forecast
    plt.plot(
        forecast_series_future.index,
        forecast_series_future.values,
        label="Forecast (SARIMA - Future)",
        color="green",
        marker="o",
    )
    plt.title("SARIMA Forecast - 6 Months Future")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    return (
        results_full if "results_full" in locals() else None,
        metrics_test,
        forecast_series_future,
    )

    """
    Fit a SARIMA(1,1,1)(1,1,1,12) model, forecast test set and future_steps.
    Plot test and future forecasts with trend lines across entire dataset
    """
    # ============================
    # Forecast Test Set
    # ============================
    try:
        model = SARIMAX(
            train_data,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit(disp=False)
        forecast_vals_test = results.forecast(steps=len(test_data))
        forecast_series_test = pd.Series(forecast_vals_test.values, index=test_index)

        # Compute metrics for test set
        metrics_test = forecast_accuracy_metrics(
            test_data.values, forecast_series_test.values
        )

        print("=== SARIMA (1,1,1)(1,1,1,12) ===")
        print("Forecasting Test Set:")
        print(f"MAE:  {metrics_test['MAE']:.2f}")
        print(f"RMSE: {metrics_test['RMSE']:.2f}")
        print(f"MAPE: {metrics_test['MAPE']:.2f}%\n")
    except Exception as e:
        print("=== SARIMA (1,1,1)(1,1,1,12) ===")
        print("SARIMA failed during test set forecasting:", e, "\n")
        forecast_series_test = pd.Series([np.nan] * len(test_data), index=test_index)
        metrics_test = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}

    # ============================
    # Plot Test Set Forecast with Trend Line
    # ============================
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data.values, label="Train", color="blue")
    plt.plot(test_data.index, test_data.values, label="Test", color="orange")
    plt.plot(
        test_index,
        forecast_series_test.values,
        label="Forecast (SARIMA - Test)",
        color="red",
        linestyle="--",
    )
    plt.title("SARIMA Forecast - Test Set")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ============================
    # Forecast Future Set
    # ============================
    try:
        # Refit SARIMA on full data
        model_full = SARIMAX(
            full_data,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results_full = model_full.fit(disp=False)
        forecast_vals_future = results_full.forecast(steps=future_steps)
        future_dates = pd.date_range(
            full_data.index[-1] + pd.offsets.MonthEnd(1), periods=future_steps, freq="M"
        )
        forecast_series_future = pd.Series(
            forecast_vals_future.values, index=future_dates
        )
    except Exception as e:
        print("SARIMA failed during future forecasting:", e)
        forecast_series_future = pd.Series(
            [np.nan] * future_steps,
            index=pd.date_range(
                full_data.index[-1] + pd.offsets.MonthEnd(1),
                periods=future_steps,
                freq="M",
            ),
        )

    # ============================
    # Plot Future Forecast with Trend Line
    # ============================
    plt.figure(figsize=(12, 6))
    plt.plot(full_data.index, full_data.values, label="Historical Demand", color="blue")

    # Compute fitted values over historical data
    fitted_full_data = results_full.predict(
        start=full_data.index[0], end=full_data.index[-1]
    )

    plt.plot(
        full_data.index, fitted_full_data.values, label="Fitted Trend", color="red"
    )
    plt.plot(
        forecast_series_future.index,
        forecast_series_future.values,
        label="Forecast (SARIMA - Future)",
        color="green",
        marker="o",
    )
    plt.title("SARIMA Forecast - 6 Months Future")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    return (
        results_full if "results_full" in locals() else None,
        metrics_test,
        forecast_series_future,
    )


def run_holt_winters(train_data, test_data, test_index, full_data, future_steps=6):
    """
    Fit Holt-Winters (additive trend, additive seasonality, period=12).
    Forecast test set and future_steps.
    Plot test and future forecasts with trend lines across entire dataset.
    """
    # ============================
    # Forecast Test Set
    # ============================
    try:
        # Fit Holt-Winters model on training data
        hw_model = ExponentialSmoothing(
            train_data, trend="add", seasonal="add", seasonal_periods=12
        )
        hw_fit = hw_model.fit()

        # Forecast for the test set
        forecast_vals_test = hw_fit.forecast(steps=len(test_data))
        forecast_series_test = pd.Series(forecast_vals_test.values, index=test_index)

        # Get in-sample fitted values for training data
        fitted_train = hw_fit.fittedvalues

        # Forecasted test set as part of the trend line
        fitted_test = forecast_series_test

        # Combine fitted_train and fitted_test for a continuous trend line
        fitted_full_test = pd.concat([fitted_train, fitted_test])

        # Compute accuracy metrics for the test set
        metrics_test = forecast_accuracy_metrics(
            test_data.values, forecast_series_test.values
        )

        print("=== Holt-Winters (Additive) ===")
        print("Forecasting Test Set:")
        print(f"MAE:  {metrics_test['MAE']:.2f}")
        print(f"RMSE: {metrics_test['RMSE']:.2f}")
        print(f"MAPE: {metrics_test['MAPE']:.2f}%\n")
    except Exception as e:
        print("=== Holt-Winters (Additive) ===")
        print("Holt-Winters failed during test set forecasting:", e, "\n")
        forecast_series_test = pd.Series([np.nan] * len(test_data), index=test_index)
        metrics_test = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
        fitted_full_test = pd.Series(
            [np.nan] * len(train_data) + [np.nan] * len(test_data),
            index=train_data.index.append(test_data.index),
        )

    # ============================
    # Plot Test Set Forecast with Trend Line
    # ============================
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data.values, label="Train", color="blue")
    plt.plot(test_data.index, test_data.values, label="Test", color="orange")
    plt.plot(
        fitted_full_test.index,
        fitted_full_test.values,
        label="Fitted Trend",
        color="red",
        linestyle="--",
    )
    plt.title("Holt-Winters Forecast - Test Set")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ============================
    # Forecast Future Set
    # ============================
    try:
        # Refit Holt-Winters model on the entire dataset (train + test)
        hw_model_full = ExponentialSmoothing(
            full_data, trend="add", seasonal="add", seasonal_periods=12
        )
        hw_fit_full = hw_model_full.fit()

        # Get fitted values for the entire historical data
        fitted_full_data = hw_fit_full.fittedvalues

        # Forecast future_steps ahead
        forecast_vals_future = hw_fit_full.forecast(steps=future_steps)
        future_dates = pd.date_range(
            full_data.index[-1] + pd.offsets.MonthEnd(1), periods=future_steps, freq="M"
        )
        forecast_series_future = pd.Series(
            forecast_vals_future.values, index=future_dates
        )
    except Exception as e:
        print("Holt-Winters failed during future forecasting:", e)
        forecast_series_future = pd.Series(
            [np.nan] * future_steps,
            index=pd.date_range(
                full_data.index[-1] + pd.offsets.MonthEnd(1),
                periods=future_steps,
                freq="M",
            ),
        )
        fitted_full_data = pd.Series([np.nan] * len(full_data), index=full_data.index)

    # ============================
    # Plot Future Forecast with Trend Line
    # ============================
    plt.figure(figsize=(12, 6))
    plt.plot(full_data.index, full_data.values, label="Historical Demand", color="blue")

    # Plot fitted trend line over historical data
    plt.plot(
        fitted_full_data.index,
        fitted_full_data.values,
        label="Fitted Trend",
        color="red",
        linestyle="--",
    )

    # Plot future forecast
    plt.plot(
        forecast_series_future.index,
        forecast_series_future.values,
        label="Forecast (Holt-Winters - Future)",
        color="green",
        marker="o",
    )
    plt.title("Holt-Winters Forecast - 6 Months Future")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    return (
        hw_fit_full if "hw_fit_full" in locals() else None,
        metrics_test,
        forecast_series_future,
    )

    """
    Fit Holt-Winters (additive trend, additive seasonality, period=12).
    Forecast test set and future_steps.
    Plot test and future forecasts with trend lines across entire dataset
    """
    # ============================
    # Forecast Test Set
    # ============================
    try:
        hw_model = ExponentialSmoothing(
            train_data, trend="add", seasonal="add", seasonal_periods=12
        )
        hw_fit = hw_model.fit()
        forecast_vals_test = hw_fit.forecast(steps=len(test_data))
        forecast_series_test = pd.Series(forecast_vals_test.values, index=test_index)

        # Compute metrics for test set
        metrics_test = forecast_accuracy_metrics(
            test_data.values, forecast_series_test.values
        )

        print("=== Holt-Winters (Additive) ===")
        print("Forecasting Test Set:")
        print(f"MAE:  {metrics_test['MAE']:.2f}")
        print(f"RMSE: {metrics_test['RMSE']:.2f}")
        print(f"MAPE: {metrics_test['MAPE']:.2f}%\n")
    except Exception as e:
        print("=== Holt-Winters (Additive) ===")
        print("Holt-Winters failed during test set forecasting:", e, "\n")
        forecast_series_test = pd.Series([np.nan] * len(test_data), index=test_index)
        metrics_test = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}

    # ============================
    # Plot Test Set Forecast with Trend Line
    # ============================
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data.values, label="Train", color="blue")
    plt.plot(test_data.index, test_data.values, label="Test", color="orange")
    plt.plot(
        test_index,
        forecast_series_test.values,
        label="Forecast (Holt-Winters - Test)",
        color="red",
        linestyle="--",
    )
    plt.title("Holt-Winters Forecast - Test Set")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ============================
    # Forecast Future Set
    # ============================
    try:
        # Refit Holt-Winters on full data
        hw_model_full = ExponentialSmoothing(
            full_data, trend="add", seasonal="add", seasonal_periods=12
        )
        hw_fit_full = hw_model_full.fit()
        forecast_vals_future = hw_fit_full.forecast(steps=future_steps)
        future_dates = pd.date_range(
            full_data.index[-1] + pd.offsets.MonthEnd(1), periods=future_steps, freq="M"
        )
        forecast_series_future = pd.Series(
            forecast_vals_future.values, index=future_dates
        )
    except Exception as e:
        print("Holt-Winters failed during future forecasting:", e)
        forecast_series_future = pd.Series(
            [np.nan] * future_steps,
            index=pd.date_range(
                full_data.index[-1] + pd.offsets.MonthEnd(1),
                periods=future_steps,
                freq="M",
            ),
        )

    # ============================
    # Plot Future Forecast with Trend Line
    # ============================
    plt.figure(figsize=(12, 6))
    plt.plot(full_data.index, full_data.values, label="Historical Demand", color="blue")

    # Compute fitted values over historical data
    fitted_full_data = hw_fit_full.fittedvalues

    plt.plot(
        full_data.index, fitted_full_data.values, label="Fitted Trend", color="red"
    )
    plt.plot(
        forecast_series_future.index,
        forecast_series_future.values,
        label="Forecast (Holt-Winters - Future)",
        color="green",
        marker="o",
    )
    plt.title("Holt-Winters Forecast - 6 Months Future")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True)
    plt.show()

    return (
        hw_fit_full if "hw_fit_full" in locals() else None,
        metrics_test,
        forecast_series_future,
    )


def main():
    # 1) Load data
    df = pd.read_csv("data/WarehouseProductDemand.csv", parse_dates=["Date"])

    # 2) Filter for Warehouse J & Category 019
    df = df[(df["Warehouse"] == "Whse_J") & (df["Product_Category"] == "Category_019")]

    # 3) Keep data only from 2012-01-01 to 2016-12-31
    start_date = pd.Timestamp("2012-01-01")
    end_date = pd.Timestamp("2016-12-31")
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    if df.empty:
        print("No data found for the specified filters and date range.")
        return

    # 4) Aggregate daily demand -> monthly
    df_daily = df.groupby("Date")["Order_Demand"].sum().reset_index()
    df_daily.set_index("Date", inplace=True)
    df_monthly = df_daily["Order_Demand"].resample("M").sum()  # <--- This is a Series
    df_monthly.name = "Demand"

    # 5) Plot the entire monthly data before any model
    plt.figure(figsize=(14, 7))
    plt.plot(df_monthly.index, df_monthly.values, marker="o", linestyle="-")
    plt.title("Monthly Demand (2012–2016) - Warehouse J, Category 019")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.grid(True)
    plt.show()

    # 6) Split into train & test (last 12 months for test)
    total_months = len(df_monthly)
    if total_months < 12:
        print("Not enough data to create a 12-month test set.")
        return

    train_end = total_months - 12
    train_data = df_monthly.iloc[:train_end]
    test_data = df_monthly.iloc[train_end:]
    test_index = test_data.index  # We'll need this to align the forecast

    # ================================
    # Run each model as a function (Forecast Test Set and Future)
    # ================================
    print("\n--- Forecasting Test Set ---\n")
    model_linear, metrics_linear_test, forecast_linear_future = run_linear_regression(
        train_data, test_data, test_index, df_monthly
    )
    model_exp, metrics_exp_test, forecast_exp_future = run_exponential_regression(
        train_data, test_data, test_index, df_monthly
    )
    model_poly, metrics_poly_test, forecast_poly_future = run_polynomial_regression(
        train_data, test_data, test_index, df_monthly
    )
    model_seasonal_only, metrics_seasonal_only_test, forecast_seasonal_only_future = (
        run_seasonality_only_regression(train_data, test_data, test_index, df_monthly)
    )
    model_seasonal, metrics_seasonal_test, forecast_seasonal_future = (
        run_seasonal_regression(train_data, test_data, test_index, df_monthly)
    )
    model_sarima, metrics_sarima_test, forecast_sarima_future = run_sarima(
        train_data, test_data, test_index, df_monthly
    )
    model_hw, metrics_hw_test, forecast_hw_future = run_holt_winters(
        train_data, test_data, test_index, df_monthly
    )


if __name__ == "__main__":
    main()
