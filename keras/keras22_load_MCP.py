# MCP = Model Check Point

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
from tensorflow.keras.models import load_model

model = load_model('./keras/CheckPoint/k21_cancer_18-0.1542.hdf5')
model.summary()
results = model.evaluate(x_test, y_test)

#3. Compile, Train
# pass

#4. Evaluate, Predict
print(results)