import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import (
    LogitResults,
)  # Correct import for LogitResults
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc

import warnings

warnings.filterwarnings("ignore")


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Parameters:
    - filepath: Path to the CSV file.

    Returns:
    - DataFrame containing the loaded data.
    """
    df = pd.read_csv(filepath)
    print(
        f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns."
    )
    return df


def encode_categorical_features(
    df: pd.DataFrame, categorical_features: list
) -> pd.DataFrame:
    """
    Encode categorical features using Label Encoding.

    Parameters:
    - df: DataFrame containing the data.
    - categorical_features: List of categorical feature names to encode.

    Returns:
    - DataFrame with encoded categorical features.
    """
    df_encoded = df.copy()
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
        print(f"Encoded '{col}' with classes: {le.classes_}")
    return df_encoded


def prepare_features(df: pd.DataFrame, features: list, outcome: str) -> tuple:
    """
    Prepare the feature matrix and target vector for modeling.

    Parameters:
    - df: DataFrame containing the data.
    - features: List of feature names to include.
    - outcome: Name of the outcome variable.

    Returns:
    - Tuple of (X, y) where X is the feature matrix with a constant added, and y is the target vector.
    """
    X = df[features].copy()
    X = sm.add_constant(X)
    y = df[outcome]
    print(f"Features and target variable '{outcome}' prepared for modeling.")
    return X, y


def fit_logistic_regression(X: pd.DataFrame, y: pd.Series) -> LogitResults:
    """
    Fit a logistic regression model.

    Parameters:
    - X: Feature matrix.
    - y: Target vector.

    Returns:
    - Fitted logistic regression model results.
    """
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)
    print("Logistic regression model fitted successfully.")
    return result


def compute_odds_ratios(result: LogitResults, feature_names: list) -> pd.DataFrame:
    """
    Compute odds ratios and log-odds from the logistic regression results.

    Parameters:
    - result: Fitted logistic regression model results.
    - feature_names: List of feature names including the constant.

    Returns:
    - DataFrame containing features, log-odds, and odds ratios.
    """
    odds_ratios = np.exp(result.params)
    log_odds = result.params
    df_results = pd.DataFrame(
        {
            "Feature": feature_names,
            "Log-Odds (Coefficient)": log_odds.values,
            "Odds Ratio": odds_ratios.values,
        }
    )
    print("Odds ratios and log-odds computed.")
    return df_results


def add_predictions(
    df: pd.DataFrame,
    model_result: LogitResults,
    X: pd.DataFrame,
    prediction_col: str = "predicted_churn",
) -> pd.DataFrame:
    """
    Add predicted probabilities to the DataFrame.

    Parameters:
    - df: Original DataFrame.
    - model_result: Fitted logistic regression model results.
    - X: Feature matrix used for prediction.
    - prediction_col: Name of the new column to store predictions.

    Returns:
    - DataFrame with an additional column for predicted probabilities.
    """
    df_with_pred = df.copy()
    df_with_pred[prediction_col] = model_result.predict(X)
    print(f"Predicted probabilities added as '{prediction_col}'.")
    return df_with_pred


def create_deciles(
    df: pd.DataFrame, prediction_col: str, decile_col: str = "decile"
) -> pd.DataFrame:
    """
    Create deciles based on the predicted probabilities.

    Parameters:
    - df: DataFrame containing predictions.
    - prediction_col: Column name of predicted probabilities.
    - decile_col: Name of the new decile column.

    Returns:
    - DataFrame with an additional decile column.
    """
    df_sorted = df.sort_values(prediction_col, ascending=False).copy()
    df_sorted[decile_col] = (
        pd.qcut(
            df_sorted[prediction_col].rank(method="first", ascending=False),
            10,
            labels=False,
        )
        + 1
    )  # Deciles 1 to 10
    print("Deciles created based on predicted probabilities.")
    return df_sorted


def compute_lift(df_sorted: pd.DataFrame, decile_col: str, y_col: str) -> pd.Series:
    """
    Compute lift for each decile.

    Parameters:
    - df_sorted: DataFrame sorted by predicted probabilities.
    - decile_col: Column name of deciles.
    - y_col: Name of the outcome variable.

    Returns:
    - Series containing lift values for each decile.
    """
    baseline_churn_rate = df_sorted[y_col].mean()
    decile_lift = df_sorted.groupby(decile_col).apply(
        lambda x: x[y_col].mean() / baseline_churn_rate
    )
    print("Lift computed for each decile.")
    return decile_lift


def plot_lift_chart(decile_lift: pd.Series) -> None:
    """
    Plot the lift chart.

    Parameters:
    - decile_lift: Series containing lift values for each decile.
    """
    plt.figure(figsize=(8, 5))
    deciles = decile_lift.index
    plt.plot(deciles, decile_lift, marker="o", linestyle="-", color="b")
    plt.xlabel("Decile")
    plt.ylabel("Lift")
    plt.title("Decile-wise Lift Chart")
    plt.xticks(deciles)
    plt.grid(True)
    plt.show()
    print("Lift chart plotted.")


def compute_cumulative_gains(df_sorted: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """
    Compute cumulative gains data.

    Parameters:
    - df_sorted: DataFrame sorted by predicted probabilities.
    - y_col: Name of the outcome variable.

    Returns:
    - DataFrame with cumulative churn and cumulative customers.
    """
    df_sorted = df_sorted.copy()
    df_sorted["cumulative_churn"] = df_sorted[y_col].cumsum() / df_sorted[y_col].sum()
    df_sorted["cumulative_customers"] = np.arange(1, len(df_sorted) + 1) / len(
        df_sorted
    )
    print("Cumulative gains data computed.")
    return df_sorted


def plot_cumulative_gains(
    df_sorted: pd.DataFrame, cumulative_customers_col: str, cumulative_churn_col: str
) -> None:
    """
    Plot the cumulative gains chart.

    Parameters:
    - df_sorted: DataFrame with cumulative gains data.
    - cumulative_customers_col: Column name for cumulative customers.
    - cumulative_churn_col: Column name for cumulative churn.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(
        df_sorted[cumulative_customers_col],
        df_sorted[cumulative_churn_col],
        label="Model",
        color="g",
    )
    plt.plot([0, 1], [0, 1], "k--", label="Random Guessing")
    plt.xlabel("Cumulative % of Customers Contacted")
    plt.ylabel("Cumulative % of Churn Captured")
    plt.title("Cumulative Gains Chart")
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Cumulative gains chart plotted.")


def display_model_summary(result: LogitResults) -> None:
    """
    Display the summary of the logistic regression model.

    Parameters:
    - result: Fitted logistic regression model results.
    """
    print(result.summary())


def display_sorted_odds_ratios(df_sorted: pd.DataFrame, title: str) -> None:
    """
    Display the sorted odds ratios and log-odds.

    Parameters:
    - df_sorted: Sorted DataFrame containing odds ratios and log-odds.
    - title: Title for the printed table.
    """
    print(f"\n=== {title} ===")
    print(df_sorted.to_string(index=False))


def main():
    # Filepath to the dataset
    filepath = "data/CustomerData_Composite-3.csv"

    # Selected optimized features
    optimized_features = [
        "total_population",
        "number_of_referrals",
        "senior_citizen",
        "city",
        "online_security",
        "online_backup",
        "premium_tech_support",
        "streaming_tv",
        "internet_type",
        "contract",
        "paperless_billing",
        "multiple_lines",
        "offer",
    ]

    # Outcome variable
    outcome_variable = "churn_value"

    # Load data
    df = load_data(filepath)

    # Identify categorical features
    categorical_features = (
        df[optimized_features].select_dtypes(include=["object"]).columns.tolist()
    )
    print(f"Categorical features to encode: {categorical_features}")

    # Encode categorical variables
    df_encoded = encode_categorical_features(df, categorical_features)

    # Prepare features and target
    X_optimized, y_optimized = prepare_features(
        df_encoded, optimized_features, outcome_variable
    )

    # Fit logistic regression model
    result_optimized = fit_logistic_regression(X_optimized, y_optimized)

    # Display model summary (optional)
    display_model_summary(result_optimized)

    # Compute odds ratios and log-odds
    df_results = compute_odds_ratios(result_optimized, X_optimized.columns.tolist())

    # Since sorting by Log-Odds and Odds Ratio results in the same order,
    # we'll display only one sorted table to avoid redundancy.

    # Sort by Log-Odds (Coefficient) descending
    df_sorted_coeff = df_results.sort_values(
        by="Log-Odds (Coefficient)", ascending=False
    )
    display_sorted_odds_ratios(
        df_sorted_coeff,
        "Odds Ratios and Log-Odds Sorted by Log-Odds (Coefficient) Descending",
    )

    # If you still want to display the second sorted table by Odds Ratio descending,
    # you can uncomment the following lines. Note that the order will be identical.

    # Sort by Odds Ratio descending
    # df_sorted_odds = df_results.sort_values(by="Odds Ratio", ascending=False)
    # display_sorted_odds_ratios(
    #     df_sorted_odds,
    #     "Odds Ratios and Log-Odds Sorted by Odds Ratio Descending",
    # )

    # Add predicted probabilities to the original DataFrame
    df_with_pred = add_predictions(df, result_optimized, X_optimized)

    # Create deciles based on predicted probabilities
    df_sorted = create_deciles(df_with_pred, "predicted_churn", "decile")

    # Compute lift per decile
    decile_lift = compute_lift(df_sorted, "decile", outcome_variable)

    # Plot Lift Chart
    plot_lift_chart(decile_lift)

    # Compute cumulative gains data
    df_cumulative = compute_cumulative_gains(df_sorted, outcome_variable)

    # Plot Cumulative Gains Chart
    plot_cumulative_gains(df_cumulative, "cumulative_customers", "cumulative_churn")


if __name__ == "__main__":
    main()
