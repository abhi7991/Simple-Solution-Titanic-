# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:10:31 2020

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:43:44 2020

@author: Abhishek.S
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:43:30 2020

@author: Abhishek.S
"""
import pandas as pd
import numpy as np
import os
os.chdir(r'D:\KAGGLE\Titanic')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
#%%

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
gender = pd.read_csv('gender_submission.csv')
#%%
data.head()
test.head()
#%%
a = data.isnull()
print(a)
#%%
sns.heatmap(a)
#%%
data.columns
data['Embarked'].unique()
#%%
b = data.groupby(['Pclass'])['Age'].mean()
print(b)
#%%Data Cleaning

#imputing age by class
def impute(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 30
        else:
            return 25
    else:
        return Age
data['Age'] = data[['Age','Pclass']].apply(impute,axis=1)
test['Age'] = test[['Age','Pclass']].apply(impute,axis=1)

##
sns.heatmap(pd.isnull(data))
sns.heatmap(pd.isnull(test))
#%%
test.head()
data.drop(['Cabin'],axis=1)
test.drop(['Cabin'],axis=1)
#%%Frature Engineering
a = pd.get_dummies(data['Sex'],drop_first=True)
a1= pd.get_dummies(test['Sex'],drop_first=True)
b= pd.get_dummies(data['Embarked'],drop_first=True)
b1 = pd.get_dummies(test['Embarked'],drop_first=True)


#%%
train = pd.concat([data,a,b],axis=1)   
test  = pd.concat([test,a1,b1],axis=1)
#%%
train.head()
test.head()

#%%
train.drop(['Sex','Embarked','Name','Ticket','Cabin'],axis=1,inplace=True)
test.drop(['Sex','Embarked','Name','Ticket','Cabin'],axis=1,inplace=True)
#test=pd.concat([test,gender])
#%%
#test.drop(['Survived'],axis=True)
#test = pd.concat([test,gender])
sns.heatmap(test.isnull())
sns.heatmap(train.isnull())
#%%

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics  import classification_report,confusion_matrix,accuracy_score,log_loss
X=train.drop(['Survived'],axis=1)
y=train['Survived']
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=10,random_state=None)
model1 = LogisticRegression()
model2 = RandomForestClassifier()
model3 = XGBClassifier()
#%%
result1 = cross_val_score(model1,X,y,cv=kfold)
result2 = cross_val_score(model2,X,y,cv=kfold)
result3 = cross_val_score(model3,X,y,cv=kfold)
print("Results_logreg:",result1)
print("Results_rfc:",result2)
print("Results_sgboost:",result3)

#%%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
#%%
xgboost = XGBClassifier()
xgboost.fit(X_train,y_train)
pred = xgboost.predict(X_test)
print('CR :',classification_report(y_test,pred))
print('CM:',confusion_matrix(y_test,pred))
print('Score:',accuracy_score(y_test,pred))
print('log_loss:',log_loss(y_test,pred))
#%%
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
pred2 = rfc.predict(X_test)
print('CR_rfc :',classification_report(y_test,pred2))
print('CM_rfc:',confusion_matrix(y_test,pred2))
print('Score_rfc:',accuracy_score(y_test,pred2))
print('log_loss_rfc:',log_loss(y_test,pred2))
#%%
model1.fit(X_train,y_train)
pred3=model1.predict(X_test)
print('CR_logreg :',classification_report(y_test,pred3))
print('CM_logreg:',confusion_matrix(y_test,pred3))
print('Score_logreg:',accuracy_score(y_test,pred3))
print('log_loss_logreg:',log_loss(y_test,pred3))



        
