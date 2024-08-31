import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np

def plot_historic_data(df: pd.DataFrame) -> None:
    """
    Plot the dataframe given in input

    Args:
        df (pd.DataFrame): Financila DataFrame to be plotted
    """

    fig, (ax1, ax2) = plt.subplots(2,1, figsize = (15,8), sharex=True)
    fig.suptitle("EUR-USD Close price with SMA(14,25,50) and RSI")

    ax1.plot(df["Close"], label="Close price")
    ax1.plot(df["SMA20"], label="SMAA20")
    ax1.plot(df["SMA50"], label="SMA50")
    ax1.plot(df["SMA200"], label="SMA200")
    ax1.set_ylabel("Close price")

    ax1.legend()
    ax1.grid(True)

    ax2.plot(df['RSI'], label='RSI', color='blue')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.axhline(y=70, color='red', linestyle='--')
    ax2.axhline(y=30, color='green', linestyle='--')
    ax2.legend()
    ax2.grid(True)

    plt.show()


def plot_seasonal_decomposition(df: pd.DataFrame, last_days: int = 100) -> None:
    """
    Plot the Seasonal decomposition of the Closing price of the input dataframe

    Args:
        df (pd.DataFrame): dataframe that contains the financial data
        last_days (int, optional): the number of values to consider of the dataframe. Defaults to 100.
    """
    plt.rcParams['figure.figsize'] = (15, 8)
    ds = df['Close'][-last_days:]
    ds.index = df.index[-last_days:]
    result = seasonal_decompose(ds, model="multiplicative", period=12)
    result.plot()
    plt.show()


def plot_seasonality(df: pd.DataFrame) -> None:
    """
    Plot the Closing price and its autocorrelation, first its original values, than its 1st differencing order
    Useful to choose then the parameter for the ARIMA model

    Args:
        df (pd.DataFrame): dataframe that contains the financial data
    """
    plt.rcParams.update({'figure.figsize':(15,8)})
    _, axis = plt.subplots(2, 2, sharex=True)

    axis[0,0].plot(df[['Close']].values)
    axis[0,0].set_title('Original Series')
    # plot the autocorrelation function
    plot_acf(df['Close'], ax=axis[0,1], lags=df.dropna().shape[0]-1)

    # 1st differencing
    axis[1,0].plot(df['Close'].diff().values)
    axis[1,0].set_title('1st order differencing')
    plot_acf( df['Close'].diff().dropna(), ax=axis[1,1], lags=df.diff().dropna().shape[0]-1)

    plt.show()

def plot_predictions(real_prices: list[np.ndarray], predicted_prices: list[np.ndarray], model_names: list[str]) -> None:
    """
    Plot real vs predicted prices for multiple models.
    
    Args:
        real_prices (list[np.ndarray]) : List of arrays containing real price data
        predicted_prices (list[np.ndarray]) : List of arrays containing predicted price data
        model_names (list[str]): List of model names
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('Real vs Predicted Prices', fontsize=16)

    for i, (real, pred, name) in enumerate(zip(real_prices, predicted_prices, model_names)):
        axes[i].plot(real, label='Real Price', color='blue')
        axes[i].plot(pred, label='Predicted Price', color='red')
        axes[i].set_title(f'{name}')
        axes[i].set_ylabel('Price')
        axes[i].legend()

    plt.show()
