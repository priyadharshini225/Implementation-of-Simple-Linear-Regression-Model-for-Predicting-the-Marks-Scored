## Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PRIYADHARSHINI S
RegisterNumber: 212223240129


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
 
Y_pred

Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:

![Screenshot 2024-08-30 114254](https://github.com/user-attachments/assets/ae333482-d5bf-4527-9182-2c8690b740f6)

![Screenshot 2024-08-30 113717](https://github.com/user-attachments/assets/170488a7-449a-4a4c-8cba-a33def2d2c5d)

![Screenshot 2024-08-30 113748](https://github.com/user-attachments/assets/a38c3460-17aa-4fb4-aad2-ab1349e61920)
![Screenshot 2024-08-30 113810](https://github.com/user-attachments/assets/58458612-339a-4251-a00c-7e0f56f596bb)
![Screenshot 2024-08-30 113837](https://github.com/user-attachments/assets/dbdba6c5-868c-4c4c-8a67-a5352ce03ac2)
![Screenshot 2024-08-30 113911](https://github.com/user-attachments/assets/702a67c3-3730-4f82-be06-35a2bb6de7b2)
![Screenshot 2024-08-30 113926](https://github.com/user-attachments/assets/2e450af3-e206-4b55-b7e4-ca87b65a9a68)
![Screenshot 2024-08-30 113943](https://github.com/user-attachments/assets/2c896797-52df-4b23-8780-47f25bd58f43)

![Screenshot 2024-08-30 140735](https://github.com/user-attachments/assets/1f07520b-d4c7-4681-a04c-d6133f20952e)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
