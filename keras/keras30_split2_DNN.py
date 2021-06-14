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



#2. Model
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.models import Model

input1 = Input(shape=(x.shape[1:]))
dense1 = Dense(64, activation='relu')(input1)
dense2 = Dense(32, activation='relu')(dense1)
dense3 = Dense(16, activation='relu')(dense2)
output1 = Dense(1)(dense3)

model = Model(inputs=input1, outputs=output1)
model.summary()

#3. Compile, Train
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=512, batch_size=1)

#4. Evaluate, Predict
result = model.evaluate(x,y)
y_pred = model.predict([[6,7,8,9,10]])
print('loss :',result)
print('pred :',y_pred)