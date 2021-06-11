# Binary Classification

import numpy as np
from sklearn.datasets import load_breast_cancer

#1. Data
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=32
)

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
model.fit(x_train, y_train, batch_size=1, epochs=128, verbose=2)

#4. Evaluate, Predict
print(model.evaluate(x_test,y_test))
print(model.predict(x_test[:5]))
print(y_test[:5])

# Execution Result
# - loss: 0.2105 - acc: 0.8947
# [0.21048222482204437, 0.8947368264198303]
# [[9.8390788e-01]
#  [9.8829204e-01]
#  [9.7964120e-01]
#  [8.3962083e-04]
#  [9.6718550e-01]]
# [1 1 1 0 1]