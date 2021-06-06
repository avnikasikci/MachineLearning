# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 18:04:41 2021

@author: Avni
"""

#region Data İmport

#%% Data İmport
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
print("-------------------------Data İmport Start-------------------------------")
data = pd.read_csv('data.csv')
heightBody = data[['heightBody']]
print(heightBody)
heightWeightBody = data[['heightBody','weightBody']]
print(heightWeightBody)
print("-------------------------Data İmport End-------------------------------")



#%% 
#%% Missing Data
print("-------------------------Missing Data Start-------------------------------")

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Age = data.iloc[:,1:4].values
print(Age)
imputer = imputer.fit(Age[:,1:4])
Age[:,1:4] = imputer.transform(Age[:,1:4])
print(Age)

country = data.iloc[:,0:1].values
print(country)

print("-------------------------Missing Data End-------------------------------")

#%% 
#%% Categories and Data Concat
print("-------------------------Categories and Data Concat Start-------------------------------")

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

country[:,0] = le.fit_transform(data.iloc[:,0])

print(country)

ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
print(country)#categories classfier tr 1 us 2 fr 0 

print(list(range(22)))
result = pd.DataFrame(data=country, index = range(22), columns = ['fr','tr','us'])
print(result)

result2 = pd.DataFrame(data=Age, index = range(22), columns = ['heightBody','weightBody','age'])
print(result2)

gender = data.iloc[:,-1].values
print(gender)

result3 = pd.DataFrame(data = gender, index = range(22), columns = ['gender'])
print(result3)

s=pd.concat([result,result2], axis=1)
print(s)

s2=pd.concat([s,result3], axis=1)
print(s2)




print("-------------------------Categories and Data Concat End-------------------------------")

#%% Test Data Learning Sperator
print("------------------------Test Data Learning Sperator Start-------------------------------")

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,result3,test_size=0.33, random_state=0)



print("------------------------Test Data Learning Sperator End-------------------------------")


#%% Data 
#data scaling
print("------------------------data scaling Start-------------------------------")
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


print("------------------------data scaling End-------------------------------")
#%% data preprocessing template
print("------------------------data preprocessing template Start-------------------------------")
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)




print("------------------------data preprocessing template End-------------------------------")

#%% data preprocessing template
#endregion
