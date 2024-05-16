import tensorflow
from tensorflow import keras 
from keras import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np 
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("Churn_Modelling.csv")
df.drop(columns = ['RowNumber','CustomerId','Surname'],inplace=True)
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first = True)
x =df.drop(columns=['Exited'])
y = df['Exited']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=1)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)


model = Sequential()
model.add(Dense(11, activation='relu', input_dim=11))
model.add(Dense(11, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(x_train_scaled, y_train, epochs = 200)
y_log = model.predict(x_test_scaled)
y_pred = np.where(y_log>0.5,1,0)

print(accuracy_score(y_test,y_pred))