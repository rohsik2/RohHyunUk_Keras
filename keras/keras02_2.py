import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. Data
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([2,4,6,8,10,12,14,16,18,20])

x_test = np.array([x for x in range(101,111)])
y_test = np.array([x+10 for x in range(101,111)])

x_predict = np.array([111,112,113])

#2. Modeling
model = Sequential()
model.add(Dense(1, input_dim=1))
# model.add(Dense(1))

#3. Compile, Train
model.compile(optimizer="adam", loss='mse')
model.fit(x=x_train, y=y_train, epochs=5000, batch_size=1)

#4. Evaluate, Predict
loss = model.evaluate(x=x_test, y=y_test, batch_size=1)
print("loss :" ,loss)

print('predict :', model.predict(x_predict))