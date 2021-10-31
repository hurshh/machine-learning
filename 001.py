import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# pandas library to convert data form csv file to numpy array
dataset = pd.read_csv(
    'dataSets/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Python/Data.csv')

# iloc class will convert whole matrix to two matrices x dependent and y independent
x = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values

from sklearn.impute import SimpleImputer

# simple imputer from sklearn will fill missing values to mean of all values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# print(x)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# this will provide hot encoding to countries
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],remainder='passthrough')
x = ct.fit_transform(x)

# [print(x)]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 1)

# print(x_test)
# print()
# print(x_train )
# print()
# print(y_test)
# print()
# print(y_train)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3: ] = sc.fit_transform(x_train[:, 3: ])
