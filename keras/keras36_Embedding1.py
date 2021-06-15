from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs =[
    'so funny', 'very nice', 'well mad movie', 
    'suggest this movie', 'want to see again', 'dumdum'
    'boring movie', 'acting bad', 'not fun', 
    'boring', 'too boring', 'very funny', 'he is handsome'
]
# Positive 1, Negative 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)

print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5) # post

print(pad_x)
print(np.unique(pad_x))
print(len(np.unique(pad_x)))

#2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D

model = Sequential()
model.add(Embedding(input_dim=28, output_dim=7, input_length=5))
model.add(LSTM(32))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3. Compile, Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, batch_size=1, epochs=128, verbose=2)

#4. Evaluate, Predict

# Data shape doesn't same -> add zeros to make same shape
# 2,4 -> 0,0,0,0,2,4
# in Sequential Data, Last Data is critical