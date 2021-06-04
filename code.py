# import packages
import numpy as np # linear algebra
import pandas as pd # read csv
import seaborn as sns # make crazy visualizations
import scipy as sp # needed for seaborn
import matplotlib.pyplot as plt # needed for seaborn
import IPython.display 

# import data sets

#train set
train_path = "https://raw.githubusercontent.com/sabrinahanke/Location-opening-restaurants/main/train.csv"
train_data = pd.read_csv(train_path)

#test set
test_path = "https://raw.githubusercontent.com/sabrinahanke/Location-opening-restaurants/main/test.csv"
test_data = pd.read_csv(test_path, index_col='Id')


# Take a first look at the data
print("Number of rows and columns of train set:")
train_data.shape()
print("Number of rows and columns of test set:")
test_data.shape()
print("Train data contains: " + str(len(train_data)) + " rows and " + str(len(train_data.columns))+ " columns")
print("Test data contains: " + str(len(test_data)) + "rows and " + str(len(test_data.columns)) + " colums")
print("First 3 rows of the train data:") 
print(train_data.head(3))
print("First 3 rows of the test data:") 
print(test_data.head(3))


# analyse train data
train_data.info() # retreiving general info about train data
train_data.dtypes # analyse data type of each column
train_data.isna().sum() # check for null entries

# disply missings
print("Missings in the train data:") 
print(train_data.isnull().sum())
print("Missing in the test_data:")
print(test_data.isnull().sum())

# data is ready to be worked with!


