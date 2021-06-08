import numpy as np

#1. Data
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255.
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255


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
        strides=1, input_shape=(32,32,3)
    )
)
model.add(
    Conv2D(
        filters=8, kernel_size=(2,2), padding='same',
        strides=1
    )
)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(10, activation='softmax'))

#3. Compile, Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, epochs=32, verbose=2, validation_split=0.2, batch_size=64)

#4. Evaluate, Predict
y_pred = model.predict(x_test)
loss = model.evaluate(x_test,y_test)
print('loss :', loss[0])
print('acc  :', loss[1])
# Execute Result when batch_size=32
# loss: 7.3356 - acc: 0.4123
# loss : 7.3355607986450195
# acc  : 0.4122999906539917