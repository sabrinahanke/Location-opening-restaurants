# import packages
import numpy # linear algebra
import pandas as pd # read csv
import seaborn as sns # make crazy visualizations
import scipy # needed for seaborn
import matplotlib.pyplot as plt # needed for seaborn
import IPython as IPy # needed for hab's vergessen und bin zu faul zum googeln
import sklearn.model_selection as sk

# import data sets

# train set
train_path = r"https://raw.githubusercontent.com/sabrinahanke/Location-opening-restaurants/main/train.csv"
train_data = pd.read_csv(train_path, index_col='Id')


# Take a first look at the data

def format():
    """
    Gives back basic info about the format of the data set
    """
    train_data.info() # retreiving general info about train data
    print('Number of rows and columns of train set:', train_data.shape)
    print('dtype: ', train_data.dtypes) # analyse data type of each column
    print('First 3 rows of the train data:', train_data.head(3))
   

# analyse train data
def check_data():
    train_data.isna().sum() # check for null entries

# disply missings
def missing_values():
    """
    Calculates sum of NAN values per column for each data set
    """
    print('Missings in the train data:', train_data.isnull().sum()) 

# data is ready to be worked with!


def juhu():
    """
    Celebrates that the data set was already pretty nice so we didn't have to do a lot lol
    """
    print('Data is all clean and ready to be worked with!')


# statistical analysis

# calculate basic statistics
def basic_statistics():
    '''Gives back basic statistical calculations'''
    train_data['revenue'].describe()


# question-specific statisitcs
# open date

# city
#train_data.groupby("City")["Id"].count()
#train_data.groupby(["City"], ["Type"])["Id"].count()

# city group

# restaurant type
#train_data.groupby("Type")["Id"].count()

# revenue
def revenue_analysis():
    print('ID of most profitable restarant:', train_data["revenue"].max) 

# visualization

# identify most, least and average profitable restaurant

# divide train_data set
def split():
    train_modified = train_data.drop(labels=['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37'], axis=1)
    train, test = sk.train_test_split(train_modified, test_size=0.25, train_size=0.75)
    print(train)
    print(test)
    return train, test

# 1. Where to open a restaurant?
def where():
    train, test = split()
    train = train.drop(lables=['Open Date','Type'])
    test = test.drop(lables=['Open Date','Type'])
    # apply machine learning
    


# 2. When to open a restaurant?
# 3. Which type is the best economic choice?


def main():
    split()
if __name__ == "__main__":
    main()