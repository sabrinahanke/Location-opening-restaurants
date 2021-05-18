# import packages
import numpy as np # linear algebra
import pandas as pd # read csv
import seaborn as sns # make crazy visualizations
import SciPy as sp # needed for seaborn
import matplotlib.pyplot as plt # needed for seaborn


# import data sets

#train set
train_path = 'https://github.com/sabrinahanke/Location-opening-restaurants/blob/8ed7de276350bc782ba89c7f32972b449f038edb/train.csv'
df_train = pd.read_csv(train_path)
df_train.head()

#test set
test_path = 'https://github.com/sabrinahanke/Location-opening-restaurants/blob/8ed7de276350bc782ba89c7f32972b449f038edb/test.csv'
df_test = pd.read_csv(test_path, index_col='Id')
df_test.head()

