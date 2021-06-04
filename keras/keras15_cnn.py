# Learning About CNN(Convolution Neural Network)


#1. Data

#2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(
	Conv2D(
		filters=10,          # Conv레이어의 Output의 개수 Dense(10)의 10에 해당함.
		kernel_size=(2,2),   # 2x2 사이즈로 그림을 자른다.
		strides=1,           # 1칸씩 움직이며 그림을 자른다.
		input_shape=(5,5,1)  # 그림의 크기
	)
)
# (4,4,10)
model.add(Conv2D(5, (2,2), padding='same')) # padding='same' doesn't change shape
# (4,4,5)
model.add(Flatten())
# (16,)
model.add(Dense(1))
model.summary()


#3. Comile, Train

#4. Evaluate, Predict