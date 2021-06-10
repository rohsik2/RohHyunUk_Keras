# LSTM Long Short Term Memory

#1. Data
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])
x = x.reshape(4,3,1)

print(x.shape, y.shape)

#2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(128, input_shape=(3, 1), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()

#3. Compile, Train
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=499, verbose=0)

#4. Evaluate, Predict
print(model.evaluate(x,y))
print(model.predict([[[5.],[6.],[7.]]]))

# 0.0002326707763131708
# [[7.977563]]