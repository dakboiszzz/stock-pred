import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from vnstock import Quote

def create_windows(X_data, y_data, n_steps=60):
    X, y = [], []
    for i in range(len(X_data) - n_steps):
        X.append(X_data[i : i + n_steps, :])
        y.append(y_data[i + n_steps])
    return np.array(X), np.array(y)

def getData(symbol = 'MBB', start_date = '2019-01-01', end_date = '2026-01-14',split_ratio = 0.8):
    """Function for getting the data from vnstock API

    Args:
        symbol (str): The ticker of the company
        start_date (str): The start date
        end_date (str): The end date
        split_ratio (float, optional): Train-test split ratio. Defaults to 0.8.

    Returns:
        X_train,y_train,X_test,y_test (np.array): Train and Test data
    """
    # Get the data from vnstock
    quote = Quote(symbol=symbol, source='VCI')
    df = quote.history(start=start_date, end=end_date, interval='1D')
    
    # Separate out some feature columns
    feature_columns = ['open', 'high', 'low', 'close', 'volume']
    data = df[feature_columns].values
    
    # Splitting train-test
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Scaling 
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    # Fit only on training data
    train_scaled_X = scaler_X.fit_transform(train_data)
    train_scaled_y = scaler_y.fit_transform(train_data[:, [3]]) # Index 3 is 'close'
    test_scaled_X = scaler_X.transform(test_data)
    test_scaled_y = scaler_y.transform(test_data[:, [3]])
    
    # Creating sliding windows
    X_train, y_train = create_windows(train_scaled_X, train_scaled_y)
    X_test, y_test = create_windows(test_scaled_X, test_scaled_y)
    

    return X_train,y_train,X_test,y_test