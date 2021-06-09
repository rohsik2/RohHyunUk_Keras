import numpy as np

#1. Data
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255


from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(
    Conv2D(
        filters=32, kernel_size=(2,2), padding='same',
        strides=1, input_shape=(28,28,1)
    )
)
model.add(
    Conv2D(
        filters=16, kernel_size=(2,2), padding='same',
        strides=1
    )
)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128))
model.add(Dense(10, activation='softmax'))

model.save('./keras/Model/k23_1_model_1.h5')

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='val_loss', patience=5, mode='min')
model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['acc']
)

hist = model.fit(
            x_train, y_train, epochs=128, 
            verbose=2, validation_split=0.2,
            callbacks=[early_stopper]
        )
model.save('./keras/Model/k23_1_model_2.h5')
