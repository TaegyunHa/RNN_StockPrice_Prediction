# RNN_StockPrice_Prediction
Using Recurrent Neural Network
This program will predict the trend of stock price (uptrend or downward)

## 1. Data preprocessing

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
X_train: np array we want to reshape
<br/>

---
## 2. Building the RNN

### Importing the libraries
```python
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
```

**from keras.models import Sequential**
- The Sequential model is a linear stack of layers. You can create a Sequential model by passing a list of layer instances to the constructor

**from keras.layers import Dense**
- The model needs to know what input shape it should expect. For this reason, the first layer in a  Sequential model needs to receive information about its input shape. 

**from keras.layers import LSTM**
- Long Short-Term Memory layer 

**from keras.layers import Dropout**
- Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.


### Initialising the RNN
```python
regressor = Sequential()
```

**Sequential()**
- Sequence of layers, predicting continuous output => regressioin (not clasification)


### Adding the first LSTM layer and some Dropout regularisation
Add LSTM regularisation and Dropout. Dropout regularisation will allow to avoid overfitting
```python
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))) #add(layer)
regressor.add(Dropout(0.2))
```
**regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))**
- units: Neurons, the number of unit - the number of LSTM cells or memory units for the LSTM layers. High demensionality requires large number  of neurons to capture complex trend. 50 Neurons were used for this experiment.
- return_sequences: True (stacked LSTM - multi-LSTM layers), False (single LSTM layer)
- input_shape: shape of input train dataset (3D - observations, time_steps, indicators/predictors) but we only need to add (2D - time_steps, indicators/predictors) since the first element (observation) will automatically taken into account. Time_steps and indicator/predictors were used for this parameter.

**regressor.add(Dropout(0.2))**
- Dropout regularisation will allow to avoid overfitting. Argument is drop out rate. Drop 20%  of layers (10 neurons) for this experiment.


### Adding the second, third, and fourth layer and Dropout regularisation
```python
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
```
- After first/initial layer, input_shape does not need to be specified because the network will automatically recognise.
- We will keep the number of neuron to deal with complexity.
- For the fourth layer, <return_sequences = False>. Output layer/value will be one since we will not need more sequences.

### Adding the output layer
We need to use full connection for output layer
```python
regressor.add(Dense(unit = 1))
```

- unit: Dimension of output layer

### Compiling the RNN
```python
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
```

- optimizer: Usually RMSprop is recommended for RNN; howerver, Adam optimizer is used for this experiment
- loss: mean sqaured error for regression problem (for classification problem, binary cross entropy will be used)


### Fitting the RNN to the Training set
```python
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
```

Different batches going into the network, generating some errors, and the error is back propagated to the neural network (Every 32 stock prices because batch_size = 32).
- input of training set (X_train): input data set
- output of training set (y_train): the ground truth
- epochs: the number of iteration to do forward/back propagation.
- batch_size: The network will be trained with batch_size for one train.
