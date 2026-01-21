#Imports
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import data_process

# Take the file path 
file_path = 'PLX_from_2019-01-01_to_2026-01-21.csv'

# Preprocessing the date
X_train, y_train, X_test, y_test, scaler = data_process(
            file_path=file_path, 
            split_ratio=0.8,
            window_size=60
        )

print(X_train.shape)
