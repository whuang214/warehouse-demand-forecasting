import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Parameters:
    - filepath: Path to the CSV file.

    Returns:
    - DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(filepath)
        print(
            f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns."
        )
        return df
    except FileNotFoundError:
        print(
            f"Error: File not found at {filepath}. Please check the path and try again."
        )
        raise
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        raise


def preprocess_data(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    """
    Select relevant features and drop missing values.

    Parameters:
    - df: Original DataFrame.
    - features: List of feature column names.
    - target: Target column name.

    Returns:
    - Cleaned DataFrame with selected features and target.
    """
    required_columns = features + [target]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following required columns are missing from the dataset: {missing_cols}"
        )

    df_clean = df[required_columns].dropna()
    print(
        f"Data preprocessed: {df_clean.shape[0]} rows remaining after dropping missing values."
    )
    return df_clean


def split_features_target(
    df: pd.DataFrame, features: List[str], target: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the DataFrame into features and target variables.

    Parameters:
    - df: Cleaned DataFrame.
    - features: List of feature column names.
    - target: Target column name.

    Returns:
    - Tuple containing:
        - X: Features DataFrame.
        - y: Target Series.
    """
    X = df[features]
    y = df[target]
    print(f"Features and target variable '{target}' separated.")
    return X, y


def scale_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize the feature variables.

    Parameters:
    - X: Features DataFrame.

    Returns:
    - Tuple containing:
        - X_scaled: Scaled feature array.
        - scaler: Fitted StandardScaler object.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Feature variables standardized.")
    return X_scaled, scaler


def split_dataset(
    X: np.ndarray, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.

    Parameters:
    - X: Scaled feature array.
    - y: Target Series.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Seed used by the random number generator.

    Returns:
    - Tuple containing:
        - X_train: Training features.
        - X_test: Testing features.
        - y_train: Training target.
        - y_test: Testing target.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Dataset split into training and testing sets with test size = {test_size}.")
    return X_train, X_test, y_train, y_test


def find_optimal_k(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    k_range: range = range(1, 30),
) -> Tuple[int, List[float]]:
    """
    Find the optimal value of k for KNN by evaluating error rates over a range of k values.

    Parameters:
    - X_train: Training features.
    - y_train: Training target.
    - X_test: Testing features.
    - y_test: Testing target.
    - k_range: Range of k values to evaluate.

    Returns:
    - Tuple containing:
        - optimal_k: The k value with the lowest error rate.
        - error_rates: List of error rates corresponding to each k.
    """
    error_rates = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        error = 1 - accuracy_score(y_test, y_pred)
        error_rates.append(error)
        print(f"k={k}: Error Rate={error:.4f}")

    optimal_k = k_range[np.argmin(error_rates)]
    print(f"Optimal k found: {optimal_k} with error rate {min(error_rates):.4f}")
    return optimal_k, error_rates


def plot_error_rates(k_range: range, error_rates: List[float]) -> None:
    """
    Plot error rates against different k values.

    Parameters:
    - k_range: Range of k values.
    - error_rates: Corresponding error rates for each k.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        k_range,
        error_rates,
        color="blue",
        linestyle="dashed",
        marker="o",
        markerfacecolor="red",
        markersize=5,
    )
    plt.title("Error Rates vs. K Values for KNN")
    plt.xlabel("K Value")
    plt.ylabel("Error Rate")
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()
    print("Error rates plotted.")


def train_final_model(
    X_train: np.ndarray, y_train: pd.Series, k: int
) -> KNeighborsClassifier:
    """
    Train the final KNN model using the optimal k.

    Parameters:
    - X_train: Training features.
    - y_train: Training target.
    - k: Optimal number of neighbors.

    Returns:
    - Trained KNeighborsClassifier model.
    """
    knn_final = KNeighborsClassifier(n_neighbors=k)
    knn_final.fit(X_train, y_train)
    print(f"Final KNN model trained with k={k}.")
    return knn_final


def evaluate_model(
    model: KNeighborsClassifier, X_test: np.ndarray, y_test: pd.Series
) -> float:
    """
    Evaluate the trained model's accuracy on the test set.

    Parameters:
    - model: Trained KNeighborsClassifier model.
    - X_test: Testing features.
    - y_test: Testing target.

    Returns:
    - Accuracy score as a float.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy:.2%}")
    return accuracy


def main():
    # Filepath to the dataset
    filepath = "data/CustomerData_Composite-3.csv"

    # Selected features and target variable
    features = [
        "age",
        "satisfaction_score",
        "cltv",
        "churn_score",
        "number_of_referrals",
    ]
    target = "churn_value"  # 0: Not Churned, 1: Churned

    # Load data
    df = load_data(filepath)

    # Preprocess data
    df_clean = preprocess_data(df, features, target)

    # Split into features and target
    X, y = split_features_target(df_clean, features, target)

    # Scale features
    X_scaled, scaler = scale_features(X)

    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X_scaled, y)

    # Find optimal k
    k_range = range(1, 30)
    optimal_k, error_rates = find_optimal_k(X_train, y_train, X_test, y_test, k_range)

    # Plot error rates vs k
    plot_error_rates(k_range, error_rates)

    # Train final model
    knn_final = train_final_model(X_train, y_train, optimal_k)

    # Evaluate final model
    final_accuracy = evaluate_model(knn_final, X_test, y_test)

    # Summary of results
    print("\n=== Summary ===")
    print(f"Optimal k: {optimal_k}")
    print(f"Final Model Accuracy: {final_accuracy:.2%}")


if __name__ == "__main__":
    main()
