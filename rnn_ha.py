# Recurrent Neural Network

# 1. Data Preprocessing
# Libraries
import numpy as np
import matplotlib.pyplot as plt #Visualising
import pandas as pd #import dataset and manage

# Importing the traning set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values 

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) # All scaled stock prices will be between 0 and 1
training_set_scaled = sc.fit_transform(training_set) # fit sc to training set, transform into scaled value (normalise)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258): # range from i-60 to i, last raws is 1257
  X_train.append(training_set_scaled[i-60:i, 0]) # when i==60, range will be 0 - 59, memorizing what's happening in 60 previous time steps to predict next value at time t+1
  y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
# RNN input shape == 3D tensor with shape (batch_size, timesteps, input_dim).
# batch_size = total number of stock prices we have, timesteop: 60, predictor(indicator))
# add samsung stock price for apple stock price since they are corelated
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # 2 dimension -> 3 dimension
# Row, cols
# 2. Building the RNN

# 3. Making the predictions and visualising the result