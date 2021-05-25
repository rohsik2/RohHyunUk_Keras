from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

#1. Data
x = np.array(range(1,101))
y = array(range(101,201))

# default shuffle=True, train_size=0.75
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.6, shuffle=True, test_size=0.4
)

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, train_size=0.5
)

#2. model
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(32))
model.add(Dense(1))

#3. compile, train
model.compile(optimizer='adam',loss='mse')
model.fit(x=x_train, y=y_train, batch_size=1, epochs=100,
            validation_data=(x_val, y_val))

#4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('loss :',loss)
print('x_predict :', [101,102,103])
result = model.predict([101,102,103])
print('y_predict :', result)
