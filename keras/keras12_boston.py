import numpy as np

#1. Data
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.engine import input_layer

dataset = load_boston()
x = dataset.data
y = dataset.target #Label

x_train, x_test, y_train, y_test = train_test_split(
                                        x,y,train_size = 0.8, random_state=66)



#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(256, input_dim=13))
model.add(Dense(512))
model.add(Dense(1024))
model.add(Dense(512))
model.add(Dense(1))

input = Input(shape =(13,))
layer1 = Dense(128)(input)
layer2 = Dense(256)(layer1)
layer3 = Dense(512)(layer2)
layer4 = Dense(256)(layer3)
output = Dense(1)(layer4)
model = Model(inputs=input, outputs=output)

#3. Compile, Train
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, batch_size=1, epochs=256,
                validation_data=(x_test, y_test), verbose=2)

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
