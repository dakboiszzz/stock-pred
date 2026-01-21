import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from vnstock import Quote

def create_windows(X_data, y_data, n_steps=60):
    X, y = [], []
    for i in range(len(X_data) - n_steps):
        X.append(X_data[i : i + n_steps, :])
        y.append(y_data[i + n_steps])
    return np.array(X), np.array(y)

def getData(symbol, start_date, end_date,split_ratio = 0.8, ):
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
    

    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    
    return train_loader, X_test_t, y_test_t