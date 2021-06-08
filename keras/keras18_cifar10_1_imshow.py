import numpy as np

#1. Data
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) # (60000, ) (10000, )
# print(x_train[0])
# print(y_train[0])


# Visualization
import matplotlib.pyplot as plt

plt.imshow(x_train[0], 'gray')
plt.show()


#2. Model

#3. Compile, Train

#4. Evaluate, Predict