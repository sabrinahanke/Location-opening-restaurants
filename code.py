# import packages
import numpy as np
import pandas as pd  # read csv
import seaborn as sns  # make crazy visualizations
import scipy  # needed for seaborn
import matplotlib.pyplot as plt  # needed for seaborn
import IPython as IPy  # needed for hab's vergessen und bin zu faul zum googeln
import sklearn.model_selection as sk
from sklearn.ensemble import RandomForestRegressor
import datetime

# import data sets

# train set
train_path = r"https://raw.githubusercontent.com/sabrinahanke/Location-opening-restaurants/main/train.csv"
train_data = pd.read_csv(train_path, index_col='Id')


# Take a first look at the data

def format():
    """
    Gives back basic info about the format of the data set
    """
    train_data.info()  # retreiving general info about train data
    print('Number of rows and columns of train set:', train_data.shape)
    print('dtype: ', train_data.dtypes)  # analyse data type of each column
    print('First 3 rows of the train data:', train_data.head(3))


# analyse train data
def check_data():
    train_data.isna().sum()  # check for null entries


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
    """
    Gives back basic statistical calculations
    """
    train_data['revenue'].describe()


# question-specific statisitcs
# open date

# city
# train_data.groupby("City")["Id"].count()
# train_data.groupby(["City"], ["Type"])["Id"].count()

# city group

# restaurant type
# train_data.groupby("Type")["Id"].count()

# revenue
def revenue_analysis():
    print('ID of most profitable restarant:', train_data["revenue"].max)


# visualization

# identify most, least and average profitable restaurant

# divide train_data set
def split():
    train_modified = train_data.drop(
        labels=['City'], axis=1)
    train, test = sk.train_test_split(train_modified, test_size=0.25, train_size=0.75)
    # print(train)
    # print(test)
    return train, test


# 1. Where to open a restaurant?
def where():
    train, test = split()
    train = train.drop(lables=['Open Date', 'Type'])
    test = test.drop(lables=['Open Date', 'Type'])

# 2. When to open a restaurant?
def when():
    '''
    extract month
    '''
    train_data['month'] = pd.DatetimeIndex(train_data['Open Date']).month
    print(train_data)

    #group by month
    df_3 = train_data[['month','revenue']].groupby('month').revenue.sum()
    print(df_3)

    #bar chart months by revenue
    divisions = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    division_average_marks = [64909366.0, 46080202.0, 43665123.0, 23630484.0, 36578002.0, 33985929.0, 27668769.0, 66011346.0, 59299412.0, 78552564.0, 56656797.0, 73095974.0]

    plt.bar(divisions, division_average_marks, color='red')
    plt.title('When is the best month to open a restaurant?')
    plt.xlabel('Months Jan-Dec')
    plt.ylabel('Revenue')
    plt.show()

    # extract month

    train_data['month'] = pd.DatetimeIndex(train_data['Open Date']).month
    print(train_data)


# 3. Which type is the best economic choice?
def type():
    '''
    plots restaurant type against average revenue
    '''
    # group by type
    df_3 = train_data[['Type','revenue']].groupby('Type').revenue.mean()
    print(df_3)

    # bar chart: type by revenue
    divisions = ['Food Court', 'In Line', 'Drive Through']
    division_average_marks = [3810007.0, 355981207.0, 250342754.0]

    plt.bar(divisions, division_average_marks, color='blue')
    plt.title('Which type makes the most revenue?')
    plt.xlabel('Types of restaurants')
    plt.ylabel('Revenue')
    plt.show()


# PREDICTION TOOL
def prep():
    '''
    divides train_data into test and training set
    '''


def random_forest():
    '''
    Random Forest Regressor
    '''
    train, test = split()

    # prepare data 
    # change data type of Open Date (str -> int/float)

    # train
    train.replace('Big Cities', 1, inplace=True)
    train.replace('Other', 0, inplace=True)
    train.replace('IL', 1, inplace=True)
    train.replace('FC', 0, inplace=True)
    train.replace('DT', 2, inplace=True)
    tempSeries1 = pd.Series(dtype=int)
    train.replace(to_replace=r'^01', value=1, regex=True, inplace=True)
    train.replace(r'^02', 2, regex=True, inplace=True)
    train.replace(r'^03', 3, regex=True, inplace=True)
    train.replace(r'^04', 4, regex=True, inplace=True)
    train.replace(r'^05', 5, regex=True, inplace=True)
    train.replace(r'^06', 6, regex=True, inplace=True)
    train.replace(r'^07', 7, regex=True, inplace=True)
    train.replace(r'^08', 8, regex=True, inplace=True)
    train.replace(r'^09', 9, regex=True, inplace=True)
    train.replace(r'^10', 10, regex=True, inplace=True)
    train.replace(r'^11', 11, regex=True, inplace=True)
    train.replace(r'^12', 12, regex=True, inplace=True)
  
    # test
    test.replace('Big Cities', 1, inplace=True)
    test.replace('Other', 0, inplace=True)
    test.replace('IL', 1, inplace=True)
    test.replace('FC', 0, inplace=True)
    test.replace('DT', 2, inplace=True)
    test.replace(r'^01', 1, regex=True, inplace=True)
    test.replace(r'^02', 2, regex=True, inplace=True)
    test.replace(r'^03', 3, regex=True, inplace=True)
    test.replace(r'^04', 4, regex=True, inplace=True)
    test.replace(r'^05', 5, regex=True, inplace=True)
    test.replace(r'^06', 6, regex=True, inplace=True)
    test.replace(r'^07', 7, regex=True, inplace=True)
    test.replace(r'^08', 8, regex=True, inplace=True)
    test.replace(r'^09', 9, regex=True, inplace=True)
    test.replace(r'^10', 10, regex=True, inplace=True)
    test.replace(r'^11', 11, regex=True, inplace=True)
    test.replace(r'^12', 12, regex=True, inplace=True)
    
    print(train.to_string())

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    X, y = make_regression(n_features=41, n_informative=2, random_state=0, shuffle=False)
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(X, y)

    print(regr.predict(train))

    # print result 
    file = open('jasminsDataframe.html', 'x')
    file.write(train.to_html())
    file.close()

def ridge ():
    '''
    Ridge Regression
    '''
    train, test = split()

    # prepare data 
    # change data type of Open Date (str -> int/float)

    # train
    train.replace('Big Cities', 1, inplace=True)
    train.replace('Other', 0, inplace=True)
    train.replace('IL', 1, inplace=True)
    train.replace('FC', 0, inplace=True)
    train.replace('DT', 2, inplace=True)
    tempSeries1 = pd.Series(dtype=int)
    train.replace(to_replace=r'^01', value=1, regex=True, inplace=True)
    train.replace(r'^02', 2, regex=True, inplace=True)
    train.replace(r'^03', 3, regex=True, inplace=True)
    train.replace(r'^04', 4, regex=True, inplace=True)
    train.replace(r'^05', 5, regex=True, inplace=True)
    train.replace(r'^06', 6, regex=True, inplace=True)
    train.replace(r'^07', 7, regex=True, inplace=True)
    train.replace(r'^08', 8, regex=True, inplace=True)
    train.replace(r'^09', 9, regex=True, inplace=True)
    train.replace(r'^10', 10, regex=True, inplace=True)
    train.replace(r'^11', 11, regex=True, inplace=True)
    train.replace(r'^12', 12, regex=True, inplace=True)
  
    # test
    test.replace('Big Cities', 1, inplace=True)
    test.replace('Other', 0, inplace=True)
    test.replace('IL', 1, inplace=True)
    test.replace('FC', 0, inplace=True)
    test.replace('DT', 2, inplace=True)
    test.replace(r'^01', 1, regex=True, inplace=True)
    test.replace(r'^02', 2, regex=True, inplace=True)
    test.replace(r'^03', 3, regex=True, inplace=True)
    test.replace(r'^04', 4, regex=True, inplace=True)
    test.replace(r'^05', 5, regex=True, inplace=True)
    test.replace(r'^06', 6, regex=True, inplace=True)
    test.replace(r'^07', 7, regex=True, inplace=True)
    test.replace(r'^08', 8, regex=True, inplace=True)
    test.replace(r'^09', 9, regex=True, inplace=True)
    test.replace(r'^10', 10, regex=True, inplace=True)
    test.replace(r'^11', 11, regex=True, inplace=True)
    test.replace(r'^12', 12, regex=True, inplace=True)
    
    print(train.to_string())
    
    from sklearn.model_selection import GridSearchCV # assess predictability
    from sklearn.linear_model import Lasso, Ridge, ElasticNet
    from sklearn.ensemble import AdaBoostRegressor
    from xgboost import XGBRegressor

    bestEstimators = []

    # define parameter



def main():
    type()
if __name__ == "__main__":
    main()
