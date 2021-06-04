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
    x, y, shuffle=True, train_size=0.8, random_state=66
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, shuffle=True, train_size=0.75, random_state=66
)

#2. Model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(256, input_shape=(30,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

#3. Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam',
                     metrics=['acc'])
cnt = 0
best_fit = 0
while(True):
    model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=2, validation_data=(x_val,y_val))
    temp_result = model.evaluate(x_test, y_test, verbose=2)
    if(best_fit < temp_result[1]):
        best_fit = temp_result[1]
        cnt = 0
    else:
        cnt += 1
    if cnt > 5:
        break
#4. Evaluate, Predict
print(model.evaluate(x_test,y_test))
print(model.predict(x_test[:5]))
print(y_test[:5])

# Execution Result
# - loss: 0.2327 - acc: 0.9298       
# [0.2327367663383484, 0.9298245906829834]   
# [[0.03133512 0.9686649 ]
#  [0.06499062 0.9350094 ]
#  [0.02586474 0.9741353 ]
#  [0.04024078 0.9597592 ]
#  [0.10198012 0.89801985]]
# [[0. 1.]
#  [0. 1.]
#  [0. 1.]
#  [0. 1.]
#  [0. 1.]]