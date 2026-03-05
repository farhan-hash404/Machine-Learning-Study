# freight_cost_prediction/DataPreprocessing/model_evaluation.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    """
    Evaluates a regression model and prints key metrics.

    Parameters:
    -----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values from the model.

    Returns:
    --------
    dict
        Dictionary containing MAE, RMSE, and R2 score.
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_score = r2_score(y_true, y_pred)

    print("Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# Example usage
if __name__ == "__main__":
    # Dummy data for testing
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([12, 18, 29, 41, 49])

    metrics = evaluate_model(y_true, y_pred)

    # freight_cost_prediction/DataPreprocessing/model_evaluation.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    """
    Evaluates a regression model and prints key metrics.

    Parameters:
    -----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values from the model.

    Returns:
    --------
    dict
        Dictionary containing MAE, RMSE, and R2 score.
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_score = r2_score(y_true, y_pred)

    print("Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# Example usage
if __name__ == "__main__":
    # Dummy data for testing
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([12, 18, 29, 41, 49])

    metrics = evaluate_model(y_true, y_pred)