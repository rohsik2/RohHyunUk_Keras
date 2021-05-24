from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

#1. Data
x = np.array(range(1,101))
y = array(range(101,201))

x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]

y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]


#2. model
model = Sequential()
model.add(Dense(64, input_dim=1))
model.add(Dense(32))
model.add(Dense(1))

#3. compile, train
model.compile(optimizer='adam',loss='mse')
model.fit(x=x_train, y=y_train, batch_size=1, epochs=50,
            validation_data=(x_val, y_val))

#4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('loss :',loss)
print('x_predict :', [101,102,103])
result = model.predict([101,102,103])
print('y_predict :', result)