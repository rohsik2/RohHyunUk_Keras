# Multi layer Perceptron
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. data
x = np.array([[1,2,3,4,5,6,7,8,9,10]
                ,[11,12,13,14,15,16,17,18,19,20]])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x = np.transpose(x)

#2. model
model = Sequential()
model.add(Dense(256, input_shape=(2,)))
model.add(Dense(1024))
model.add(Dense(1))

#3. compile, train
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, batch_size=1, epochs=128)

#4. evaluate, predict
loss = model.evaluate(x, y)
print(loss)
x_predict = [[11,12,13],[21,22,23]]
x_predict = np.transpose(x_predict)
result = model.predict(x_predict)
print(result)