import numpy as np

#1. Data
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target

# One Hot Encoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=32
)

#2. Model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(4,))
h1 = Dense(256)(input1)
h2 = Dense(512)(h1)
h3 = Dense(256)(h2)
output1 =  Dense(3, activation='softmax')(h3)
model = Model(inputs=input1, outputs=output1)

#3. Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=1, epochs=64, verbose=2)

#4. Evaluate, Predict
results = model.evaluate(x_test, y_test)
print('results : ', results)
y_predict = model.predict(x_test)
y_predict2 = []
for i in range(len(y_predict)):
    y_predict2.append([])
    for j in range(len(y_predict[i])):
        if max(y_predict[i]) == y_predict[i][j]:
            y_predict2[-1].append(1)
        else:
            y_predict2[-1].append(0)