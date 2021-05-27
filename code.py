# import packages
import numpy as np # linear algebra
import pandas as pd # read csv
import seaborn as sns # make crazy visualizations
import scipy as sp # needed for seaborn
import matplotlib.pyplot as plt # needed for seaborn

# import data sets
# 1. Download the csv file from GitHub account, read and turn it into a pandas dataframe

#train set
train_path = "https://raw.githubusercontent.com/sabrinahanke/Location-opening-restaurants/main/train.csv"
df_train = pd.read_csv(train_path)
df_train.head()

#test set
test_path = "https://raw.githubusercontent.com/sabrinahanke/Location-opening-restaurants/main/test.csv"
df_test = pd.read_csv(test_path, index_col='Id')
df_test.head()

## Checking for  the number of rows and columns in the dataset using pandas
print(f"Number of rows :{df_train.shape[0]} \nNumber of columns:{df_train.shape[1]}")

#retreiving info about data
df_train.info()

# analyse data type of each column
df_train.dtypes

# check for null entries 
df_train.isna().sum()

# visualization
# 1. Where to open a restaurant?
# 2. When to open a restaurant?
# 3. Which type is the best economic choice? (FC, DT, IL)
# test

