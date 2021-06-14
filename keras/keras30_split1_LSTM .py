import numpy as np

#1. Data

size = 6
org_data = np.array(range(1,11))
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    aaa = np.array(aaa)
    x = np.array(aaa[:, :-1])
    y = np.array(aaa[:, -1])
    return x, y
x, y = split_x(org_data, size)

x = x.reshape(5, 5, 1)

#2. Model
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.models import Model, Sequential

model = Sequential()
model.add(LSTM(32, input_shape=(5, 1), activation='relu', return_sequences=True))
model.add(LSTM(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. Compile, Train
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=128, batch_size=1)

#4. Evaluate, Predict
result = model.evaluate(x,y)
x_pred = np.array([6,7,8,9,10])
x_pred = x_pred.reshape(1,5,1)
y_pred = model.predict(x_pred)
print('loss :',result)
print('pred :',y_pred)