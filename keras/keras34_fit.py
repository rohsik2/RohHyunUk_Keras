# Image Data Generator

from re import S
import numpy as np

#1. Data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    './tmp/horse-or-human',
    target_size=(300,300),
    batch_size=10000,
    class_mode='binary', # y data is binary data(yes or no)
)

# Foun 1027 images belonging to 2 classes
xy_test = test_datagen.flow_from_directory(
    './tmp/testdata',
    target_size=(300,300),
    batch_size=10000,
    class_mode='binary',
)
x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

#2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPool2D

model = Sequential()
model.add(Conv2D(16, (3,3), 1, 'same', input_shape=(300,300,3)))
model.add(Conv2D(16, (3,3), 1, 'same'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), 2))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3,3), 1, 'same'))
model.add(Conv2D(32, (3,3), 1, 'same'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), 2))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), 1, 'same'))
model.add(Conv2D(64, (3,3), 1, 'same'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), 2))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), 1, 'same'))
model.add(Conv2D(128, (3,3), 1, 'same'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), 2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.4))
model.add(Dense(256))
model.add(Dropout(0.4))
model.add(Dense(128))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

#3. Compile, Train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, vebose=2, epochs=64, validation_data=(x_test,y_test))

#4. Evaluate, Predict
results = model.evaluate(x_test, y_test)
print(results)
