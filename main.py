import data_fetcher
from data_exploration import plot_historic_data, plot_seasonal_decomposition, plot_seasonality, plot_predictions
from arima_model import get_arima_forecast
from lstm_model import get_lstm_forecast
from svr_model import get_svr_forecast
import matplotlib.pyplot as plt
from evaluation import evaluate_model

def main():

    # get dataframe
    df = data_fetcher.get_dataframe()

    # data exploration and analysis
    plot_historic_data(df)

    plot_seasonal_decomposition(df)

    plot_seasonality(df)

    # implement the ARIMA model
    print("### ARIMA MODEL ###")
    close_price_real_arima, forecast_arima = get_arima_forecast(df)

    # implement the lstm model
    print("\n\n### LSTM MODEL ###")
    close_price_real_lstm, forecast_lstm = get_lstm_forecast(df)

    # implement the SVR
    print("\n\n### SVR ###")
    close_price_real_svr, forecast_svr =  get_svr_forecast(df)

    # evaluate the model
    metrics_arima = evaluate_model(close_price_real_arima, forecast_arima, "ARIMA")

    metrics_lstm = evaluate_model(close_price_real_lstm, forecast_lstm, "LSTM")

    metrics_svr = evaluate_model(close_price_real_svr, forecast_svr, "SVR")  

    models = ["ARIMA", "LSTM", "SVR"]
    mae_values = [metrics_arima[0], metrics_lstm[0], metrics_svr[0]]
    rmse_values = [metrics_arima[2], metrics_lstm[2], metrics_svr[2]]
    mape_values = [metrics_arima[3], metrics_lstm[3], metrics_svr[3]]

    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.bar(models, mae_values)
    plt.title('Mean Absolute Error')
    plt.subplot(132)
    plt.bar(models, rmse_values)
    plt.title('Root Mean Squared Error')
    plt.subplot(133)
    plt.bar(models, mape_values)
    plt.title('Mean Absolute Percentage Error')
    plt.tight_layout()
    plt.show()

    real_prices = [close_price_real_arima, close_price_real_lstm, close_price_real_svr]
    predicted_prices = [forecast_arima, forecast_lstm, forecast_svr]

    plot_predictions(real_prices, predicted_prices, models)


if __name__ == "__main__":
    main()