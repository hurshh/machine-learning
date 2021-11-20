import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('dataSets/Part 2 - Regression/Section 5 - Multiple Linear Regression/Python/50_Startups.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = ct.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# print(x_train)
# print()
# print(y_train)

# predicting result

y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), (y_test.reshape(len(y_pred), 1))), 1))

# getting values for custom profit

print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

# get equation for linear regression

print(regressor.coef_)
print(regressor.intercept_)