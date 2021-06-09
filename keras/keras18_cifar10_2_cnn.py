import numpy as np

#1. Data
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.layers.merge import Average

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255.
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255.


from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D, BatchNormalization, AveragePooling2D

model = Sequential()
model.add(
    Conv2D(
        filters=16, kernel_size=(3,3), strides=1,
        padding='same', input_shape=(32,32,3), activation='relu'
    )
)
model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
model.add(Dropout(rate=0.3))

model.add(Conv2D(16, (3,3), strides=1,padding='same', activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=1, padding='same'))
model.add(Dropout(rate=0.3))

model.add(Conv2D(16, (3,3), strides=1,padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=1, padding='same'))
model.add(Dropout(rate=0.3))

model.add(Conv2D(32, (3,3), strides=1,padding='same', activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=1, padding='same'))
model.add(Dropout(rate=0.3))




model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(10, activation='softmax'))

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='val_loss', patience=5, mode='min')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(
        x_train, y_train, epochs=128, callbacks=[early_stopper],
        verbose=2, validation_split=0.2, batch_size=32
    )

#4. Evaluate, Predict
y_pred = model.predict(x_test)
loss = model.evaluate(x_test,y_test)
print('loss :', loss[0])
print('acc  :', loss[1])
# Execute Result 
# loss : 0.901841402053833
# acc  : 0.7052000164985657