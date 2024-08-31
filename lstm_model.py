import pandas as pd
import numpy as np
from utils import FEATURES, feature_normalization, plot_overall_close_w_prediction
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


features = FEATURES

SEQUENCE_LENGTH = 60

NUM_EPOCHS = 1 #50

# define the LSTM model
class LSTMModel(nn.Module):
    """
    LSTM model class, needed for the training loop
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size) -> None:
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # initialize hidden state (h0) and cell state (c0) with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # get the last output of the lstm and pass it to the fully connected layer

        return out


def _create_sequence(data: pd.DataFrame, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create the sequence independent values (X) and the correspondi depending values (y)
    So are created two lists:
    * X (independent variable) of type -> List[List[float]]
    * y (dependent variable) of type -> List[float]

    Args:
        data (DataFrame): dataframe with all the data
        sequence_length (int): the length of the sequence used to predict the dependent variable (y)

    Returns:
        np.array, np.array: X (independent variable, all the features values and close price of last "sequence length"), y (dependent variable, 'Close' price)
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length].values)
        y.append(data.iloc[i+sequence_length]['Close'])

    return np.array(X), np.array(y)


def _get_data_loader_datasets_and_test_tensors(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """
    Create the tensors for each variable (dependent and independent) for each dataset (train, validation, test), and create the dataloader, needed to train the model

    Args:
        X_train (np.ndarray)
        X_val (np.ndarray)
        X_test (np.ndarray)
        y_train (np.ndarray)
        y_val (np.ndarray)
        y_test (np.ndarray)

    Returns:
        tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]: Return the first two Dataloader for the training phase, and return the Tensors of the test set, for the prediction phase
    """
    # convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val).view(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).view(-1, 1)

    # Create DataLoader for training and validation
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, X_test, y_test


def _get_lstm_trained(model: LSTMModel, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim, criterion: nn.MSELoss) -> LSTMModel:
    """
    Train the LSMT model and evaluate it

    Args:
        model (LSTMModel)
        train_loader (DataLoader)
        val_loader (DataLoader)
        optimizer (torch.optim)
        criterion (nn.MSELoss)

    Returns:
        LSTMModel: model already trained
    """
    # Training loop
    for epoch in range(NUM_EPOCHS):
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad() # clear old gradients
            loss.backward() # calculate new gradients
            optimizer.step() # update model parameters

        # Validation
        model.eval()
        with torch.no_grad(): # disable gradient calsulation
            val_loss = 0
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()

            val_loss /= len(val_loader) # avg validation loss across all batches 

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

    return model


def _evaluate_model(model: LSTMModel, criterion: nn.MSELoss, X_test: torch.FloatTensor, y_test: torch.FloatTensor) -> np.ndarray:
    """
    Evaluate the model, given the test set get the predictions

    Args:
        model (LSTMModel)
        criterion (nn.MSELoss)
        X_test (torch.FloatTensor)
        y_test (torch.FloatTensor)

    Returns:
        np.ndarray: predictions of the LSTM model
    """
    # Evaluation
    model.eval()
    with torch.no_grad(): # disable gradient calculation
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')

    # Make predictions
    predictions = test_outputs.numpy()

    return predictions


def get_lstm_forecast(df:pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize the dataframe, create the sequences and subdivide them in train, validation and test, then forecast the testing dataset

    Args:
        df (pd.DataFrame): financial dataframe

    Returns:
        tuple[np.ndarray, np.ndarray]: the actual closing price and the predicted one
    """
    # split data in training validation and test
    train_df, val_test_df = train_test_split(df, test_size=0.3, shuffle=False)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5, shuffle=False)

    # scale the data
    # doing this to correctly inverse_transform at the end
    train_df_scaled, _ = feature_normalization(train_df, features, MinMaxScaler())
    val_df_scaled, _ = feature_normalization(val_df, features, MinMaxScaler())
    test_df_scaled, test_scaler_dict = feature_normalization(test_df, features, MinMaxScaler())

    df_scaled = pd.concat([train_df_scaled,val_df_scaled, test_df_scaled])

    # create the sequences (X,y)
    X, y = _create_sequence(df_scaled, SEQUENCE_LENGTH)

    # split the data
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, shuffle=False)

    train_loader, val_loader, X_test, y_test = _get_data_loader_datasets_and_test_tensors(X_train, X_val, X_test, y_train, y_val, y_test)

    # instantiate the model
    input_size = len(df.columns)
    hidden_size = 50
    num_layers = 2
    output_size = 1

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    _get_lstm_trained(model, train_loader, val_loader, optimizer, criterion)

    predictions = _evaluate_model(model, criterion, X_test, y_test)

    # inverse transformation
    predictions = test_scaler_dict['Close'].inverse_transform(predictions.reshape(-1, 1)).flatten()
    y_test_actual = test_scaler_dict['Close'].inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
    
    # plot the predictions
    plot_overall_close_w_prediction("LSTM" ,df['Close'][:len(y_train)], df['Close'][len(y_train) + len(y_val):], predictions, df['Close'][len(y_train): len(y_train)+len(y_val)])

    return y_test_actual, predictions




    