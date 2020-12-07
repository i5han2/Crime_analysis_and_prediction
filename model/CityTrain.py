import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import train_test_split
import re
import numpy as np
import pickle
import joblib

df = pd.read_csv('datasets/Crime v5.csv')
df = df.drop(['Latitude','Longitude'],axis='columns')
inputs = df.drop('City Id',axis='columns')
target = df['City Id']

X_train, X_test, Y_train, Y_test = train_test_split(inputs, target,test_size = 0.20)

# len(X_train)

model = xgb.XGBClassifier(n_jobs=-1,random_state=42,n_estimators=70,learning_rate=0.9)
model.fit(X_train.values,Y_train.values)

with open("CityPredict.pkl", 'wb') as f_out:
    pickle.dump(model, f_out) # write final_model in .bin file
    f_out.close()  # close the file