from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from utils import FEATURES, feature_normalization, plot_overall_close_w_prediction
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

features = FEATURES


def _init_arima_model(train_data: pd.DataFrame, y_train: pd.Series) -> ARIMAResults:
    """
    inintialize the ARIMA model and fit it

    Args:
        train_data (pd.DataFrame): the training features (independent variable)
        y_train (pd.Series): the value that depend on the training features (dependent variable)

    Returns:
        ARIMAResults: the ARIMA model already fitted 
    """
    model = ARIMA(y_train.values, order=(5,1,1), exog=train_data[features].reset_index(drop=True))
    model_fit = model.fit()

    return model_fit


def get_arima_forecast(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize the dataframe, subdivide it in train and test, and initialize the ARIMA model, then forecast the values of the test set and return the real close price and the predicted one

    Args:
        df (pd.DataFrame): financial dataframe

    Returns:
        tuple[np.ndarray, np.ndarray]: the actual closing price and the predicted one
    """
    # split data in training validation and test
    train_df, test_df = train_test_split(df, test_size=0.15, shuffle=False)

    # scale the data
    train_df_scaled, _ = feature_normalization(train_df, features, StandardScaler())
    test_df_scaled, test_scaler_dict = feature_normalization(test_df, features, StandardScaler())

    df_scaled = pd.concat([train_df_scaled, test_df_scaled])

    X = df_scaled[features][:-1]
    y = df_scaled['Close'].shift(-1)[:-1]

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    
    model = _init_arima_model(X_train, y_train)

    prediction_scaled = model.forecast(steps=len(y_test), exog=X_test)

    # normalize the values
    predictions = test_scaler_dict['Close'].inverse_transform(prediction_scaled.values.reshape(-1, 1)).flatten()
    y_test_actual = test_scaler_dict['Close'].inverse_transform(y_test.values.reshape(-1, 1)).flatten()


    plot_overall_close_w_prediction("ARIMA", df['Close'][:len(y_train)], df['Close'][-len(y_test):], predictions)

    return y_test_actual, predictions





