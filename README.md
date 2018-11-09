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
- .values - Only the values in the DataFrame will be returned, the axes labels will be removed.

```python

```
```python

```
```python

```
```python

```
```python

```
```python

```
