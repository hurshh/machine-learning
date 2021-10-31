import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('dataSets/Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train , x_test, y_train, Y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train, regressor.predict(x_train),color = 'blue')
plt.title('salary vs experience(training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

plt.scatter(x_test,Y_test,color = 'red')
plt.plot(x_test, regressor.predict(x_test),color = 'blue')
plt.title('salary vs experience(test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()





