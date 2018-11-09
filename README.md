# RNN_StockPrice_Prediction
Using Recurrent Neural Network
This program will predict the trend of stock price (uptrend or downward)

## Data preprocessing

### Importing the libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

**numpy:** Processing the dataset such as array or matrices\
**matplotlib.pyplot:**  Visualising the dataset\
**pandas:** Import dataset and manage it

### [Google_Stock_Price_Train.csv]
Before explaining the function that used, let's see the contents of the csv file.

| Date      | Open   | High   | Low    | Close  | Volume      |
|-----------|--------|--------|--------|--------|-------------|
| 1/3/2017  | 778.81 | 789.63 | 775.8  | 786.14 | "1,657,300" |
|    ...    |   ...  |   ...  |   ...  |   ...  |     ...     |
| 1/31/2017 | 796.86 | 801.25 | 790.52 | 796.79 | "2,160,600" |


### Importing the training set
Importing the data from csv file to train the RNN. Imported data will be stored in the numpy array.

```python
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values
```

**dataset_train = pd.read_csv('<csv_filename.csv>')**
- To import the dataset from the csv, read_csv method of pandas can be used. size (1258, 6) DataFrame Type np.array is generated and stored in 'dataset_train' variable.

**training_set = dataset_train.iloc[:,1:2].values**
- traning set is the input data which RNN will be trained.
- .iloc[rows,cols] - Indexing and Selecting Data. All rows are used whereas only second colum is used as an input dataset.
Important point to note is that **[:,1]** cannot be used since we are generating numpy array not vector.
To avoid this instead putting single index, put **[:,1:2]** (2 is excluded as the property of python).
- .values - Only the values in the DataFrame will be returned, the axes labels will be removed. Convert DataFrame type into flaot.


### Feature Scaling
Standardisation vs Normalisation\
For the sigmoid function as the activation function for the output layer of RNN, Normalisation is recommended

```python
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
```
**sc = MinMaxScaler(feature_range = (0, 1))**
- call the MinMaxScaler object. All scaled stock prices will be between 0 and 1

**training_set_scaled = sc.fit_transform(training_set)**
- Normalisation - fit sc to training set, transform into scaled value. Preserve the original dataset


### Creating a data structure with 60 timesteps and 1 output
60 time steps - at each time t, the RNN is going to look at 60 stock prices before time t, that is stock prices between 60 days before time t and time t. Based on captured in during the 60 previous time steps it will try to predict next output. 60 time steps are the best information from which RNN is trying to learn and understand correlation and trends. Based on its understanding, it's going to predict next output, the stock price at time t+1

```python
X_train = []
y_train = []
for i in range(60, 1258):
  X_train.append(training_set_scaled[i-60:i, 0])
  y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
```

**X_train = [], y_train = []**
- Create 2 seprate entities. X_train(input): 60 stock prices, y_train(output): the stock price on the next day

**for i in range(60, 1258):**
- range from 60 to 1257, the last raw dataset is 1257. Since this network predict the value at the next time step based on previous 60 time steps, i needs to starts from 60 (values between 0 ~ 59 time step will be used for first prediction at the 60 time steop)

**X_train.append(training_set_scaled[i-60:i, 0])**
- When i == 60, range will be 0 - 59, memorizing what's happening in 60 previous time steps to predict next value at time t+1

**y_train.append(training_set_scaled[i, 0])**
- ground truth to compare with predicted value. Based on difference, weight will be updated

**X_train, y_train = np.array(X_train), np.array(y_train)**
- Convert it as matrix from array. Xtrain y axis == financial day (from 60 to last day), x axis == all input 60 stock price to predict the value for the next time step


### Reshaping
**RNN input shape**
3D tensor with shape (batch_size, timesteps, input_dim).
- batch_size = total number of stock prices we have
- timesteps: 60
- input_dim: predictor(indicator) - Add samsung stock price for apple stock price since they are corelated
  - more indicator will make the model more robust

```python
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
```

**X_train = np.reshape(X_train, (y_axis<row/stock_prices>, x_axis<col/time_steps>, indicator_num))**
- X_train.shape[0]: y_axis, X_train.shape[1]: x_axis
X_train: np array we want to reshape\


```python

```
```python

```
```python

```