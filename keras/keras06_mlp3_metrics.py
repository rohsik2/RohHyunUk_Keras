import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. Data
# Below Data doesn't have any meaning
# dummy data for modeling
x = np.array([[100, 85, 70], [90, 85, 100], 
              [80, 50, 30], [43, 60, 100]])
y = np.array([75, 65, 33, 85])


#2. Model
model = Sequential()
model.add(Dense(256, input_shape=x.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dense(128))
model.add(Dense(1))

#3. Compile, Train
model.compile(optimizer='adam', loss='mse', metrics=['mse','mae','accuracy'])
model.fit(x, y, batch_size=1, epochs=200)
loss = model.evaluate(x, y)

#4. Evaluate, Predict
print(loss)
