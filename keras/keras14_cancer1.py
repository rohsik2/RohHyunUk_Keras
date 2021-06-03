# Binary Classification

import numpy as np
from sklearn.datasets import load_breast_cancer

#1. Data
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

#2. Model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(256, input_shape=(30,), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. Compile, Train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x, y, batch_size=1, epochs=128, verbose=2, validation_split=0.2)

#4. Evaluate, Predict
print(model.evaluate(x,y))
print(model.predict(x[-5:-1]))
print(y[-5:-1])

# Execution Result
#  - loss: 0.1334 - acc: 0.9473
# [0.13339120149612427, 0.9472759366035461]
# [[5.2238724e-10]
#  [7.6416896e-08]
#  [4.6309233e-03]
#  [5.8071925e-09]]
# [0 0 0 0]