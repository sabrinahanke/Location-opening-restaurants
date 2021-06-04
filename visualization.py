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

# visualization

# identify most, least and average profitable restaurant
print("ID of most profitable restarant:")
print(train_data["revenue"].max) 

# 1. Where to open a restaurant?
# print("Number of opened restaurants per citytype:")
# print(train_data.groupby("Type"))

# 2. When to open a restaurant?
# 3. Which type is the best economic choice?