from sklearn.model_selection import train_test_split
from utils import FEATURES, feature_normalization, plot_overall_close_w_prediction
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

features = FEATURES

def get_svr_forecast(df:pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize the dataframe, subdivide it in train and test, and initialize the SVR model, then forecast the values of the test set and return the real close price and the predicted one

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


    # Create and train the SVR model
    svr = SVR(kernel='rbf', C=100, epsilon=0.1)
    svr.fit(X_train, y_train)

    prediction_scaled = svr.predict(X_test)

    y_test_actual = test_scaler_dict['Close'].inverse_transform(y_test.values.reshape(-1, 1)).ravel()
    prediction = test_scaler_dict['Close'].inverse_transform(prediction_scaled.reshape(-1,1)).ravel()

    plot_overall_close_w_prediction("SVR", df['Close'][:len(y_train)], df['Close'][-len(y_test):], prediction)

    return y_test_actual, prediction
    
