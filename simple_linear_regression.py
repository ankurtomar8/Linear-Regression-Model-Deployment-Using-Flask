import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os 
import pickle




dataset = pd.read_csv(r"/Path to dataset /Salary_Data.csv")

X = dataset.iloc[ : , :-1].values

y = dataset.iloc[ : , -1].values

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X , y ,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)

print(regressor.score(X_test,y_test))

y_pred = regressor.predict(X_test)



#visualization of training set 

plt.scatter(X_train , y_train , color ="red")
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training Sets)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualization of test set 

plt.scatter(X_test , y_test , color ="red")
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Test Sets)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


with open('LRmodel_pickle','wb') as f:
    pickle.dump(regressor,f)




print(regressor.predict([[12]])) # Takes input as 2D array 
print(regressor.coef_)
print(regressor.intercept_)


#Final Equation Will be 
# y = 9312.57512673 * Years of experience + 26780.099150628186
#
#



