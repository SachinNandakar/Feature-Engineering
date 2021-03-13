import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

### Encoding Nominal Categorical variables ###
print("Encoding Nominal Categorical variables")
# using get_dummies method (using pandas class) #
print("using get_dummies method")
data = { 'Age':[18,20,23,19,22],  'City':['A','B','C','B','A'], 'Win':['Yes','No','Yes','Yes','No']}
df = pd.DataFrame(data)
print(df)

# List different categories in each attribute/column
print("Categories in each attribute column")
for col in df.columns:
    print(col, ":", len(df[col].unique() ) )

# get_dummies method & implementing drop_first parameter (False/True)
df_dummy = pd.get_dummies(df, drop_first=False)
df_dummy_drop = pd.get_dummies(df, drop_first=True)
print(df_dummy)
print(df_dummy.shape)
print(df_dummy_drop)
print(df_dummy_drop.shape)

# One-hot encoding Method (using sklearn package) #
ohe = OneHotEncoder(handle_unknown='error')
cat_col = ['City','Win']
encoded = ohe.fit_transform(df[cat_col])
df_encoded = pd.DataFrame(encoded, index=df.index)
# concat the encoded data columns with original data columns
concat_data = pd.concat([df, df_encoded], axis=1)
print(concat_data)
#NOTE: The first method of get_dummies is easy & better compared to OHE method


### Encoding Ordinal Categorical variables ###
print("\nEncoding Ordinal Categorical variables")
# Label encoding method
data_mg = { 'Marks': [78,56,87,91,45,62],  'Grade': ['B','C','A','A','D','B']}
df_mg = pd.DataFrame(data_mg)
print(df_mg)

le = LabelEncoder()
df_gtf = le.fit_transform(df_mg['Grade'])
df_gt = pd.DataFrame(df_gtf, index=df_mg.index)
Concat_data = pd.concat([df_mg, df_gt], axis=1)
print("After Label Encoding\n", Concat_data)

# Using Simple one line code Label Encoding #
df_mg['Grade_Code'] = LabelEncoder().fit_transform(df_mg['Grade'])
print(df_mg)


### Find & Replace method (if limited number of categories) ###
print("\n Find & Replace method")
data_child = { 'MaleChild': ['One','Two','One','One','Three'],  'FemaleChild': ['Two','One','Three','Two','One'] }
df_child = pd.DataFrame(data_child)
print(df_child)
# Here, we can use the dictionary to replace the string with numerical
cat_text_num = { 'MaleChild': {'One':1, 'Two':2, 'Three':3},
                             'FemaleChild': {'One':1, 'Two':2, 'Three':3} }
df_child = df_child.replace(cat_text_num)
print("\nAfter Replacement\n",df_child)


###  Encoding using OrdinalEncoder method ###
print("\nEncoding using OrdinalEncoder method")
car_data = { 'make': ['Tata','MS','Tata','Hyundai','Mahindra','MS'] }
df_car = pd.DataFrame(car_data)
print(df_car)
ord_enc = OrdinalEncoder()
df_car['Make_code'] = ord_enc.fit_transform(df_car[['make']] )
print("After OrdinalEncoding\n", df_car)

### Best Simulation of all methods with one dataset integrated way ###
# Refer this link: https://pbpython.com/categorical-encoding.html #
