# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
Program to implement the simple linear regression model For predicting the marks scored.

Developed by: G.Lutheesh
RegisterNumber:  212221230029
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/student_scores.csv')
df.head()
df.tail()
```
## Segregating data to variables
```
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
```
## Displaying predicted values
```
y_pred
```
## Displaying actual values
```
y_test
```
## Graph plot for training data
```
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
```
## Graph plot for test data
```
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_test,regressor.predict(x_test),color="yellow")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()


mse=mean_squared_error(y_test,y_pred)
print('MSE= ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE =',mae)

import numpy as np
rmse=np.sqrt(mse)
print('RMSE =',rmse)
```
## Output:
## df.head()
![Screenshot 2023-08-25 112729](https://github.com/Lutheeshgoparapu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94154531/d37f2671-874b-4e2a-951c-997a94070c94)
## df.tail()
![Screenshot 2023-08-25 112737](https://github.com/Lutheeshgoparapu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94154531/c5b4cf18-9bb3-4a16-927d-d8701e56f77c)

## Array value of X
![Screenshot 2023-08-25 113120](https://github.com/Lutheeshgoparapu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94154531/0ac95284-62a5-4d10-a387-356b46cdcb22)
## Array value of Y
![Screenshot 2023-08-25 113134](https://github.com/Lutheeshgoparapu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94154531/9e220368-5f97-4839-93bd-d26fab937f0a)
## Values of Y prediction
![Screenshot 2023-08-25 113404](https://github.com/Lutheeshgoparapu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94154531/485bc887-d311-4dba-acf6-ab54ae73d0e3)
## Values of Y test
![Screenshot 2023-08-25 113409](https://github.com/Lutheeshgoparapu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94154531/47aed11e-00a7-46ae-bb5c-5322a2617e48)
## Training Set Graph
![Screenshot 2023-08-25 113425](https://github.com/Lutheeshgoparapu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94154531/d46e3135-8f36-4d96-b5f2-0d59a788924a)

## Test Set Graph
![Screenshot 2023-08-25 114413](https://github.com/Lutheeshgoparapu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94154531/875ba1b5-2d81-4b41-8e69-2f37823f45dd)

## Values of MSE, MAE and RMSE
![image-2](https://github.com/Lutheeshgoparapu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/94154531/c6649215-070a-4546-85a7-419f3e87234c)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
