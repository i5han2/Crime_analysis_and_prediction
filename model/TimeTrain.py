import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import train_test_split
import re
import numpy as np
import pickle
import joblib

df = pd.read_csv('datasets/Crime v4.1.csv')
inputs = df.drop('Time Id',axis='columns')
target = df['Time Id']

X_train, X_test, Y_train, Y_test = train_test_split(inputs, target,test_size = 0.20)

# len(X_train)

model = xgb.XGBClassifier(n_jobs=-1,random_state=42,n_estimators=1000,learning_rate=0.55)
model.fit(X_train.values,Y_train.values)

# with open("TimeCategory.pkl", 'wb') as f_out:
#     pickle.dump(model, f_out) # write final_model in .bin file
#     f_out.close()  # close the file

print(accuracy_score(Y_train,model.predict(X_train.values)))
print(accuracy_score(Y_test,model.predict(X_test.values)))
print(model.predict(np.array([[2, 1, 7, 2, 0, 30, -97]])))
print(model.predict(np.array([[9, 0, 7, 2, 5, 34, -118]])))
print(model.predict(np.array([[2, 1, 2, 5, 4, 39, -104]])))
print(model.predict(np.array([[9, 0, 4, 0, 1, 39, -76]])))

# 0.9313413515991587
# 0.5882099131654172
# [3]
# [0]
# [3]
# [0]