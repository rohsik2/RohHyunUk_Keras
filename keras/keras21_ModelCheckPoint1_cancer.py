import numpy as np
from sklearn.datasets import load_breast_cancer

#1. Data
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, shuffle=True, train_size=0.75, random_state=66
)

#2. Model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(256, input_shape=(30,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopper = EarlyStopping(monitor='val_loss', patience=5, mode='min')
modelpath = './keras/CheckPoint/k21_cancer_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(
        filepath=modelpath, monitor='val_loss', save_best_only=True,
        mode='auto'
    )
model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['acc']
)

hist = model.fit(
            x_train, y_train, batch_size=1, epochs=128, 
            verbose=2, validation_data=(x_val, y_val),
            callbacks=[early_stopper, cp]
        )

#4. Evaluate, Predict
print(model.evaluate(x_test,y_test))
print(model.predict(x_test[:5]))
print(y_test[:5])

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('losses')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train_loss', 'val_loss'])
plt.show()
# Execution Result
