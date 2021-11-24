import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(
    'dataSets/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Python/Position_Salaries.csv')

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1)

# feature scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# print(x)
# print(y)

from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(x, y)

print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]]))))