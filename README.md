# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

# AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

# Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm
1. Import pandas as pd.
2. Obtain the information of the data.
3. Print the sum of null datas.
4. Using DecsionTreeClassifier imported from sklearn we get the accuracy.
5. Print the predicted values.

# Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Akshayaa M 
RegisterNumber: 212222230009
*/

import pandas as pd
data=pd.read_csv("/content/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

# Output:
## data.head()
![Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn](1.png)
## data.info()
![Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn](2.png)
## data.isnull().sum()
![Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn](3.png)
## Left column value count
![Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn](5.png)
## data.head() for salary
![Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn](6.png)
## x.head()
![Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn](4.png)
## Accuracy value
![Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn](7.png)
## Data prediction
![Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn](8.png)

# Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
