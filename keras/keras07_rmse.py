# Multi layer Perceptron
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


#1. data
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [11,12,13,14,15,16,17,18,19,20]])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x = np.transpose(x)

#2. model
model = Sequential()
model.add(Dense(256, input_shape=(2,)))
model.add(Dense(1024))
model.add(Dense(1))

#3. compile, train
model.compile(optimizer='adam', loss='mse', metrics=['acc','mae'])
model.fit(x, y, batch_size=1, epochs=128)

#4. evaluate, predict
y_predict = model.predict(x)
loss = model.evaluate(x,y)
print('MSE :', mean_squared_error(y, y_predict))
print("RMSE :",RMSE(y, y_predict))
print('MAE :', loss[2])
