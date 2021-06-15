import numpy as np

#1. Data
import numpy as np

x = np.array([
        [1, 2, 3], [2, 3, 4], [3, 4, 5], 
        [4, 5, 6], [5, 6, 7], [6, 7, 8], 
        [7, 8, 9], [8, 9, 10], [9, 10, 11], 
        [10, 11, 12], [20, 30, 40], [30,40,50],
        [40, 50, 60]
    ])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])
x = x.reshape(13, 3, 1)

print(x.shape, y.shape)

#2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape

model = Sequential()
model.add(LSTM(32, input_shape=(3, 1), activation='relu', return_sequences=True))
model.add(LSTM(32, input_shape=(3, 1), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))

model.summary()

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='loss', patience=15, mode='min')

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=499, verbose=2, callbacks=[early_stopper])

#4. Evaluate, Predict
x_pred = np.array([50,60,70])
x_pred = x_pred.reshape(1,3,1)
print(model.evaluate(x,y))
print(model.predict(x_pred))
# Execution result
# if predict value > 75 n < 85 -> Good Tuning
# 0.308273583650589
# [[79.47302]]