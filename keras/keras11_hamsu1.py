# Functional Model

import numpy as np
from sklearn.model_selection import train_test_split


#1. Data
x = np.array([range(100), range(301,401), range(1,101), range(100), range(301,401)])
y = np.array([range(711,811), range(1,101), range(201,301)])

x = np.transpose(x)
y = np.transpose(y)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
                                        x,y,train_size = 0.8, random_state=66)


#2. Model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(3, input_dim=5))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(3))
model.summary()

input1 = Input(shape=(5,))
dense1 = Dense(3)(input1)
dense1_1 = Dense(7)(dense1)
dense2 = Dense(4)(dense1_1)
output1 = Dense(3)(dense2)

model = Model(inputs=input1, outputs=output1)
model.summary()


'''
#3. Compile, Train
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, batch_size=1, epochs=64,
                validation_data=(x_test, y_test),verbose=2)

#4. Evaluate, Predict
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
import tensorflow as tf

y_predict = model.predict(x_test)
loss = model.evaluate(x_test, y_test)
print('MSE :', mean_squared_error(y_test, y_predict))
print("RMSE :",RMSE(y_test, y_predict))
print('R2 :',r2_score(y_test, y_predict))
'''