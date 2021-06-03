import numpy as np

#1. Data
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target

# One Hot Encoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y[48:52])
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

#2. Model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(4,))
h1 = Dense(1000)(input1)
h2 = Dense(100)(h1)
h3 = Dense(256)(h2)
h4 = Dense(5)(h3)
output1 =  Dense(3, activation='softmax')(h4)
model = Model(inputs=input1, outputs=output1)

#3. Compile, Train
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, batch_size=5, epochs=100, verbose=2)

#4. Evaluate, Predict
results = model.evaluate(x_test, y_test)
print('results : ', results)

y_predict = model.predict(x_test)
print("input: ",x_test[:5])
print("GT: ", y_test[:5])
print("predict: ", y_predict[:5])