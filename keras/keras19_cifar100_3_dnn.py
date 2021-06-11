from tensorflow.keras.datasets import cifar100

#1. Data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(60000, 32*32*3)
y_train = y_train.reshape(60000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)
y_test = y_test.reshape(10000, 32*32*3)

#2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
model = Sequential()
model.add(Dense(2048, input_shape=(32*32*3,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. Compile, Train 53491016126025
from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='val_loss', patience=10, mode='min')

model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy',
    metrics=['acc']
)
model.fit(
    x_train,y_train, batch_size=32, epochs=64, verbose=2,
     validation_split=0.2
)

#4. Evaluate, Predict
result = model.evaluate(x_test, y_test)
print("loss :", result[0])
print("acc  :", result[1])

y_pred = model.predict(x_test[3])
print("predict :", y_pred[0])
print("answer  :", y_test[3])


import matplotlib.pyplot as plt
plt.imshow(x_test[3], 'gray')
plt.show()

# Execute Result
# - loss: 4.6055 - acc: 0.0100
# loss : 4.605493068695068
# acc  : 0.009999999776482582