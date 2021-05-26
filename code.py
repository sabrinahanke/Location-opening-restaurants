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

# visualization
# 1. Where to open a restaurant?
# 2. When to open a restaurant?
# 3. Which type is the best economic choice? (FC, DT, IL)
# test

