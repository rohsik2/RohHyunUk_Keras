import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. Data
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])

x_test = np.array([6,7,8])
y_test = np.array([6,7,8])

#2. Modeling
model = Sequential()
model.add(Dense(64, input_dim=1))
model.add(Dense(1))

#3. Compile, Train
model.compile(optimizer="adam", loss='mse')
model.fit(x=x_train, y=y_train, epochs=500, batch_size=1)

#4. Evaluate, Predict
loss = model.evaluate(x=x_test, y=y_test, batch_size=1)
print("loss :" ,loss)
print('predict :', model.predict([9]))