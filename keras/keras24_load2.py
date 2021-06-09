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
from tensorflow.keras.models import load_model
modelpath='./keras/Model/k23_1_model_2.h5'
model = load_model(filepath=modelpath)

model.summary()

#3. Compile, Train
# skip

#4. Evaluate, Predict
results = model.evaluate(x_test, y_test)
print(results)
