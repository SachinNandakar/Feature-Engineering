import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

data_frame = pd.DataFrame()
data_frame = pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
                   "toy": [np.nan, 'Batmobile', 'Bullwhip'],
                   "born": [pd.NaT, pd.Timestamp("1940-04-25"),
                            pd.NaT]})


### Detect missing values ###
print("\nDetecting missing values of Dataframe")
print(data_frame.isna())
print("\nSum of missing values in each attribute")
print(data_frame.isnull().sum() )
## NOTE: both methods returns the boolean value indiacting true for missing value


### Detect Existing Values ###
print("\nDetecting Existing Values of Dataframe")
print(data_frame.notna().sum() )
#print(data_frame.notnull() )


### Delete the entire row of missing value ###
print("\nBefore Delete / Drop\n", data_frame)
data_frame.dropna(axis=0, inplace=True)
print("\nAfter Delete / Drop\n", data_frame)

## Redifine data_frame to original shape ##
data_frame = pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
                   "toy": [np.nan, 'Batmobile', 'Bullwhip'],
                   "born": [pd.NaT, pd.Timestamp("1940-04-25"),
                            pd.NaT]})


### Fill NA/NaN values using specified method ###
data_frame.fillna(0)
print("\nFill NA(0)\n", data_frame)
data_frame.fillna(method='ffill')
print(data_frame)


### Using Imputer class to input the missing values to other known values ###
data = np.array([[1, np.nan, 2], [2, 3, np.nan], [-1, 4, 2]])
print(data)
imp = SimpleImputer(strategy='mean')
print(" \nAfter imputing mean values")
print( imp.fit_transform(data) )

imp_mf = SimpleImputer(strategy='most_frequent')
data_fill_med = imp_mf.fit_transform(data)
print(" \nAfter imputing mode value")
print(data_fill_med)

print("\nBefore Filling Missing Data\n", data_frame)
imp_con = SimpleImputer(strategy='constant', fill_value="Val_Fill")
print("\nAfter Filling Missing Data\n", imp_con.fit_transform(data_frame) )
