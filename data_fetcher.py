import pandas as pd
from utils import FILE_NAME

# add some features
def _calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """
    SMA -> Simple Moving Average
        = (a_1, ..., a_p)/p

    Returns:
        Series: Simple Moving Average of a specified period of time
    """
    sma = data.rolling(window=period).mean()
    
    return sma


def _calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    RSI -> Relative Strength Index
        = 100 - (100/ 1 + RS)
        where RS (Relative Strength) = avg_gain/avg_loss

    Returns:
        Series: RSI of the Close price
    """

    # calculate price changes
    delta = data['Close'].diff()

    # separate gains and losses
    gain = delta.where(delta>0, 0)
    loss = -delta.where(delta<0, 0)

    # calculate the average on gain and loss
    avg_gain = _calculate_sma(gain, period=period)
    avg_loss = _calculate_sma(loss, period=period)

    # calculate the RS (relative strength)
    rs = avg_gain/avg_loss
    
    rsi = 100 - (100 / (1 + rs))

    return rsi

def get_dataframe() -> pd.DataFrame:
    """
    Read the csv file and compose the dataframe

    Returns:
        DataFrame: with the following columns: Close, RSI, SMA20, SMA50, SMA200
    """

    df = pd.read_csv(FILE_NAME, parse_dates=['Date'])

    # choose the Close price
    df = df[['Date', 'Close']].dropna()

    rsi = _calculate_rsi(df)
    sma_20 = _calculate_sma(df['Close'], period=20)
    sma_50 = _calculate_sma(df['Close'], period=50)
    sma_200 = _calculate_sma(df['Close'], period=200)

    # add elements to the dataframe
    df['RSI'] = rsi
    df['SMA20'] = sma_20
    df['SMA50'] = sma_50
    df['SMA200'] = sma_200

    df = df.dropna().reset_index(drop=True) 
    df.set_index(['Date'], inplace=True)

    return df

