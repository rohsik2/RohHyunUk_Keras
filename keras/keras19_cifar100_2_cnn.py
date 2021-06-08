from tensorflow.keras.datasets import cifar100
import numpy as np

#1. Data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

layer = preprocessing.Normalization()
layer.adapt(x_train)
x_train = layer(x_train)

layer = preprocessing.Normalization()
layer.adapt(x_test)
x_test = layer(x_test)

#2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(
    Conv2D(
        filters=128, kernel_size=(2,2), strides=1,
        padding='same', input_shape=(32,32,3)
    )
)
model.add(
    Conv2D(
        filters=64, kernel_size=(2,2), strides=1,
        padding='same', input_shape=(32,32,3)
    )
)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='val_loss', patience=10, mode='min')

model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy',
    metrics=['acc']
)
model.fit(
    x_train,y_train, batch_size=32, epochs=128, verbose=2,
    validation_split=0.2
)

#4. Evaluate, Predict
result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc  :", result[1])

y_pred = model.predict(x_test[2:3])
y_pred = np.where(y_pred>=y_pred.max())
print("predict :", y_pred)
print("answer  :", y_test[3])


import matplotlib.pyplot as plt
plt.imshow(x_test[3], 'gray')
plt.show()
# loss: 4.4749 - acc: 0.1760
# loss : 4.4749436378479
# acc  : 0.17599999904632568