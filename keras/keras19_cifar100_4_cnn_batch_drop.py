from tensorflow.keras.datasets import mnist
import numpy as np

#1. Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
'''
layer = preprocessing.Normalization()
layer.adapt(x_train)
x_train = layer(x_train)

layer = preprocessing.Normalization()
layer.adapt(x_test)
x_test = layer(x_test)
'''
x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

#2. Model
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation

model = Sequential()

model.add(BatchNormalization(input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(256, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(256, (2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))



model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(512,  activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(100, activation='softmax'))

model.summary()

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='val_loss', patience=15, mode='min')

model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy',
    metrics=['acc']
)
model.fit(
    x_train,y_train, batch_size=64, epochs=128, verbose=2,
    validation_split=0.2, callbacks=[early_stopper]
)

model.save(filepath='./keras/Model/k29_cifar100.h5')

#4. Evaluate, Predict
result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc  :", result[1])

y_pred = model.predict(x_test[2:3])
y_pred = np.where(y_pred>=y_pred.max())
print("predict :", y_pred)
print("answer  :", y_test[3])


# loss : 2.009660243988037
# acc  : 0.5254999995231628

