from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def calculate_mape(real: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error
    Insicate how far the predictions are off on average

    Args:
        real (np.ndarray): _description_
        predicted (np.ndarray): _description_

    Returns:
        int: _description_
    """
    return np.mean(np.abs((real - predicted)/real))*100

def evaluate_model(real: np.ndarray, predicted: np.ndarray, model_name: str) -> tuple[float, float, float, float]:
    """
    Evaluate the model, calcualte the errors on the predictions done 
    - MAE (Mean Absolute Error): Average of absolute differences between predicted and actual values.
    - MSE (Mean Squared Error): Average of squared differences between predicted and actual values.
    - RMSE (Root Mean Squared Error): Square root of MSE, interpretable in original data units.
    - MAPE (Mean Absolute Percentage Error): Average of absolute percentage differences between predicted and actual values.

    Args:
        real (np.ndarray): array of real close prices
        predicted (np.ndarray): array of predictions of the close price
        model_name (str): Name of the model

    Returns:
        tuple[float, float, float, float]: mae, mape, rmse, mape
    """
    mae = mean_absolute_error(real, predicted)
    mse = mean_squared_error(real, predicted)
    rmse = np.sqrt(mse)
    mape = calculate_mape(real, predicted)

    print(f"Evaluation metrics for {model_name}:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print()

    return mae, mse, rmse, mape