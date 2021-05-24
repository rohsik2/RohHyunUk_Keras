import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. Data
x_train = np.array([1,2,3,4,5,6,7,14,15,16])
y_train = np.array([1,2,3,4,5,6,7,14,15,16])

x_test = np.array([9,10,11])
y_test = np.array([9,10,11])

#2. Model
model = Sequential()
model.add(Dense(8, input_dim=1, activation='relu'))
model.add(Dense(64))
model.add(Dense(1))

#3. Compile, Train
model.compile(optimizer='adam', loss='mse')
model.fit(x=x_train, y=y_train, batch_size=1, epochs=500,
#                validation_data=(x_val, y_val))
            validation_split=0)
#4. Evaluate, Predict
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

result = model.predict([19])
print('predict :',result)