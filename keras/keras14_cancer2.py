# Binary Classification

import numpy as np
from sklearn.datasets import load_breast_cancer

#1. Data
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

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
model.add(Dense(2, activation='softmax'))

#3. Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=1, epochs=64, verbose=2)

#4. Evaluate, Predict
print(model.evaluate(x_test,y_test))
print(model.predict(x_test[:5]))
print(y_test[:5])

# Execution Result
# - loss: 0.3199 - acc: 0.8684
# [0.3198934495449066, 0.8684210777282715]
# [[8.2145520e-02 9.1785449e-01]
#  [7.0860900e-02 9.2913914e-01]
#  [2.4050921e-02 9.7594905e-01]
#  [9.9999952e-01 4.7987635e-07]
#  [1.4624098e-01 8.5375899e-01]]
# [1 1 1 0 1]