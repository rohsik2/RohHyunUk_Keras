import numpy as np
import tensorflow as tf

#1. Data
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30, input_dim=1))  # input's dimension is one == one input node
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

#3. Compile, Traine
model.compile(loss='mse', optimizer='adam')
model.fit(x=x, y=y, epochs=500, batch_size=1)

#4. Evaluate, Predict
loss = model.evaluate(x,y, batch_size=1)
results = model.predict([4])
print("loss :", loss)
print("results :", results)