import pandas as pd
import sklearn.preprocessing
import matplotlib.pyplot as plt
import numpy as np

FILE_NAME = "EURUSD=X.csv"

FEATURES = ['Close', 'RSI', 'SMA20', 'SMA50', 'SMA200']

def plot_overall_close_w_prediction(model: str, y_train: pd.Series, y_test: pd.Series, forecast: np.ndarray, y_val:pd.Series = pd.Series()) -> None:
    """
    Plot the Close price values, of the training set (when present) and testing set, with the forecasting values on top of it

    Args:
        y_train (pd.Series)
        y_test (pd.Series)
        forecast (np.ndarray)
    """

    plt.figure(figsize=(12, 6))
    plt.title(f"Overview Historical Close price and {model} Forecasting over the test dataset")
    plt.plot(y_train, label='Train')
    if not y_val.empty:
        plt.plot(y_val, label='Validation')
    plt.plot(y_test, label='Test')
    plt.plot(y_test.index[-len(forecast):], forecast, label='Forecast')
    plt.legend()
    plt.show()

# normalize the features
def feature_normalization(df: pd.DataFrame, features: list, scaler_input: sklearn.preprocessing) -> tuple[pd.DataFrame, dict]:
    """
    Normalize all the features of the dataframe in given in input using a scaler

    Args:
        df (pd.DataFrame): dataframe containing the financial data
        features (list): list of features to normalize of the input dataframe
        scaler_input (sklearn.preprocessing): the scaler function that normalize the values

    Returns:
        tuple[pd.DataFrame, dict]: dataframe normalized and the dictionary that for each column contain the Normalizer already fitted, this is useful for the inverse transformation
    """
    scalers_dict = {}

    for f in features:
        scaler = scaler_input
        df[f] = scaler.fit_transform(df[f].values.reshape(-1,1))

        scalers_dict[f] = scaler
    
    return df, scalers_dict