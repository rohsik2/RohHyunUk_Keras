import numpy as np

#1. Data
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, x_test) (60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test) (60000, ) (10000, )
# print(x_train[0])
# print(y_train[0])


# Visualization
import matplotlib.pyplot as plt

plt.imshow(x_train[0], 'gray')
plt.show()


#2. Model

#3. Compile, Train

#4. Evaluate, Predict