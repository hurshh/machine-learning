import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('dataSets/Part 2 - Regression/Section 5 - Multiple Linear Regression/Python/50_Startups.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])],remainder='passthrough')
x = ct.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


