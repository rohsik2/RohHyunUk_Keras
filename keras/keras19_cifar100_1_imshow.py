from tensorflow.keras.datasets import cifar100

#1. Data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#1.1 Visualization
import matplotlib.pyplot as plt

plt.imshow(x_train[0], 'gray')
plt.show()