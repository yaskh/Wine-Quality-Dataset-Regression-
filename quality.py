#Importing the libraries 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score 

#Importing the dataset
dataset = pd.read_csv("winequality-red.csv",delimiter = ';')
X = dataset.iloc[:,0:10].values
y = dataset.iloc[:, -1].values

#Seperating the dataset 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)


#Fitting the regressor to the training set 
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the values
y_pred = regressor.predict(X_test)

r2_score(y_test,y_pred)


#Using Support Vector Regression 
from sklearn.svm import SVR
regressor = SVR(kernel = 'poly',degree =3)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
r2_score(y_pred,y_test)



#Using Deicision Tree regression 
regressor = DecisionTreeRegressor()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
r2_score(y_pred,y_test)
