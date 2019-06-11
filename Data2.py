import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv');
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=10,random_state=0)

# fitting linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# prediction the test results
y_pred = regressor.predict(x_test)

# visualising the Training set results
plt.scatter(x_train, y_train,color="yellow")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experence')
plt.ylabel('Salary')
plt.show()

# visualising the Test set results
plt.scatter(x_test, y_test,color="yellow")
plt.plot(x_test, y_pred,color="green")
plt.title('Salary vs Experience (TestSet)')
plt.xlabel('Years of Experence')
plt.ylabel('Salary')
plt.show()