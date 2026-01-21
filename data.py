import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def create_windows(X_data, y_data, n_steps=60):
    X, y = [], []
    for i in range(len(X_data) - n_steps):
        X.append(X_data[i : i + n_steps, :])
        y.append(y_data[i + n_steps])
    return np.array(X), np.array(y)
def data_process(file_path, split_ratio=0.8, window_size=60):
    
    df = pd.read_csv(file_path)
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
    X_train, y_train = create_windows(train_scaled_X, train_scaled_y, window_size)
    X_test, y_test = create_windows(test_scaled_X, test_scaled_y, window_size)
    

    return X_train,y_train,X_test,y_test,scaler_y