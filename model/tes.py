import xgboost
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('datasets/Crime v4.1.csv')
# df = df.drop(['Latitude','Longitude'],axis='columns')
inputs = df.drop('Time Id',axis='columns')
target = df['Time Id']

X_train, X_test, Y_train, Y_test = train_test_split(inputs, target,test_size = 0.20)
#
# df = df.drop(['Latitude','Longitude'],axis='columns')
#
# inputs = df.drop('City Id',axis='columns')
# target = df['City Id']

with open('TimeCategory.pkl', 'rb') as f_in:
    model2 = pickle.load(f_in)
# print(df.describe())
# print(model2.predict(np.array(inputs.iloc[67000].values).reshape(1,-1)))
print(accuracy_score(Y_train,model2.predict(X_train.values)))
print(accuracy_score(Y_test,model2.predict(X_test.values)))
# print(model2.predict(np.array([[2, 1, 7, 2, 0, 30, -97]])))
print(model2.predict(np.array([[2, 1, 7, 2, 0, 30, -97]])))
print(model2.predict(np.array([[9, 0, 7, 2, 5, 34, -118]])))
print(model2.predict(np.array([[2, 1, 2, 5, 4, 39, -104]])))
print(model2.predict(np.array([[9, 0, 4, 0, 1, 39, -76]])))
# crimeType: 86.78
# time:86.25
# city:56.78