# Location-opening-restaurants

## File descriptions
train.csv - the training set. Use this dataset for training your model. 
test.csv - the test set. To deter manual "guess" predictions, Kaggle has supplemented the test set with additional "ignored" data. These are not counted in the scoring.
sampleSubmission.csv - a sample submission file in the correct format

## Data fields
Id : Restaurant id. 
Open Date : opening date for a restaurant
City : City that the restaurant is in. Note that there are unicode in the names. 
City Group: Type of the city. Big cities, or Other. 
Type: Type of the restaurant. FC: Food Court, IL: Inline, DT: Drive Thru, MB: Mobile
P1, P2 - P37: There are three categories of these obfuscated data. Demographic data are gathered from third party providers with GIS systems. These include population in any given area, age and gender distribution, development scales. Real estate data mainly relate to the m2 of the location, front facade of the location, car park availability. Commercial data mainly include the existence of points of interest including schools, banks, other QSR operators.
Revenue: The revenue column indicates a (transformed) revenue of the restaurant in a given year and is the target of predictive analysis. Please note that the values are transformed so they don't mean real dollar values. 

## Data Info
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 137 entries, 0 to 136
Data columns (total 43 columns):
     Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   Id          137 non-null    int64
 1   Open Date   137 non-null    object
 2   City        137 non-null    object
 3   City Group  137 non-null    object
 4   Type        137 non-null    object
 5   P1          137 non-null    int64
 6   P2          137 non-null    float64
 7   P3          137 non-null    float64
 8   P4          137 non-null    float64
 9   P5          137 non-null    int64
 10  P6          137 non-null    int64
 11  P7          137 non-null    int64
 12  P8          137 non-null    int64
 13  P9          137 non-null    int64
 14  P10         137 non-null    int64
 15  P11         137 non-null    int64
 16  P12         137 non-null    int64
 17  P13         137 non-null    float64
 18  P14         137 non-null    int64
 19  P15         137 non-null    int64
 20  P16         137 non-null    int64
 21  P17         137 non-null    int64
 22  P18         137 non-null    int64
 23  P19         137 non-null    int64
 24  P20         137 non-null    int64
 25  P21         137 non-null    int64
 26  P22         137 non-null    int64
 27  P23         137 non-null    int64
 28  P24         137 non-null    int64
 29  P25         137 non-null    int64
 30  P26         137 non-null    float64
 31  P27         137 non-null    float64
 32  P28         137 non-null    float64
 33  P29         137 non-null    float64
 34  P30         137 non-null    int64
 35  P31         137 non-null    int64
 36  P32         137 non-null    int64
 37  P33         137 non-null    int64
 38  P34         137 non-null    int64
 39  P35         137 non-null    int64
 40  P36         137 non-null    int64
 41  P37         137 non-null    int64
 42  revenue     137 non-null    float64
dtypes: float64(9), int64(30), object(4)
memory usage: 46.1+ KB
Number of rows :137 
Number of columns:43
