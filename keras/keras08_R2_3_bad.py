# Multi layer Perceptron
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
import tensorflow as tf


from sklearn.metrics import r2_score


#1. data
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])

x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])

x_pred = np.array([16,17,18])


#2. model
model = Sequential()
model.add(Dense(1024, input_dim=1))
model.add(Dense(1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))


#3. compile, train
model.compile(optimizer='adam', loss='mae')
model.fit(x_train, y_train, batch_size=1, epochs=100)

#4. evaluate, predict
y_predict = model.predict(x_test)
loss = model.evaluate(x_test, y_test)
print('MSE :', mean_squared_error(y_test, y_predict))
print("RMSE :",RMSE(y_test, y_predict))
print('R2 :',r2_score(y_test, y_predict))
